import os
import math
import random
import numpy as np
from PIL import Image
import pyvista as pv
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2

base_path_3d = "/media/sadiq252/6E70-6785/3D"
test_base_2d = "/media/sadiq252/6E70-6785/2D"
labels = ['camera', 'guitar', 'laptop', 'piano', 'bottle']
os.makedirs("result", exist_ok=True)
os.makedirs("result/visual_compare", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

EMBED_DIM = 4096
VIEW_DIM = 3

AZ_STEP = 45
EL_STEP = 45
ROLL_STEP = 45

REGRESSOR_EPOCHS = 1000
BATCH_SIZE = 1000
LR = 0.001

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.backbone = resnet

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class AngleRegressor(nn.Module):
    def __init__(self, embedding_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, emb):
        return self.net(emb)


embedding_net = EmbeddingNet().to(device)
angle_regressor = AngleRegressor().to(device)

for p in embedding_net.parameters():
    p.requires_grad = False

embedding_net.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

embedding_bank = defaultdict(list)

def render_object_and_extract_embeddings(obj_file, label, az_step=AZ_STEP, el_step=EL_STEP, roll_step=ROLL_STEP):
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background('white')
    plotter.window_size = (224, 224)
    try:
        plotter.import_obj(obj_file)
    except Exception as e:
        print("Failed to import:", obj_file, e)
        plotter.close()
        return

    bounds = plotter.renderer.bounds
    plotter.view_xy()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    r = max(xmax - xmin, ymax - ymin, zmax - zmin) * 3.5

    idx = 0
    for roll in range(0, 360, roll_step):
        for el in range(0, 180, el_step):
            for az in range(0, 360, az_step):
                rad_az = math.radians(az)
                rad_el = math.radians(el)
                rad_roll = math.radians(roll)

                plotter.camera.azimuth = az
                plotter.camera.elevation = el
                plotter.camera.roll = roll
                plotter.render()

                img_array = plotter.screenshot(return_img=True, transparent_background=True)
                img = Image.fromarray(img_array).convert("RGB")

                out_dir = f"result/{label}/az_el_roll"
                os.makedirs(out_dir, exist_ok=True)
                img_path = os.path.join(out_dir, f"{label}_roll{roll}_el{el}_az{az}.jpg")
                img.save(img_path)

                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = embedding_net(img_tensor).cpu().numpy()[0]

                embedding_bank[label].append({
                    'emb': emb,
                    'az': float(az),
                    'el': float(el),
                    'roll': float(roll),
                    'img_path': img_path
                })
                idx += 1
    plotter.close()


MAX_OBJS_PER_CLASS = 10
for label in labels:
    class_path = os.path.join(base_path_3d, label)
    obj_files = [os.path.join(root, f)
                 for root, _, files in os.walk(class_path)
                 for f in files if f == "model_normalized.obj"]
    obj_files = sorted(obj_files)[:MAX_OBJS_PER_CLASS]
    for obj_file in obj_files:
        print(f"Processing {label}: {obj_file}")
        render_object_and_extract_embeddings(obj_file, label)

total_renders = sum(len(v) for v in embedding_bank.values())
print("Total rendered views stored:", total_renders)
if total_renders == 0:
    raise SystemExit("No rendered views found. Check base_path_3d and .obj files.")

class EmbAngleDataset(Dataset):
    def __init__(self, embedding_bank):
        self.items = []
        for label, lst in embedding_bank.items():
            for d in lst:
                emb = d['emb'].astype(np.float32)
                az = math.radians(d['az'])
                el = math.radians(d['el'])
                roll = math.radians(d['roll'])
                target = np.array([az, el, roll], dtype=np.float32)
                self.items.append((emb, target))
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        emb, tgt = self.items[idx]
        return emb, tgt

dataset = EmbAngleDataset(embedding_bank)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Regressor dataset size:", len(dataset))

optimizer = torch.optim.Adam(angle_regressor.parameters(), lr=LR)
criterion = nn.MSELoss()

angle_regressor.train()
for epoch in range(REGRESSOR_EPOCHS):
    epoch_loss = 0.0
    for batch_emb, batch_tgt in train_loader:
        batch_emb = batch_emb.to(device)
        batch_tgt = batch_tgt.to(device)

        pred = angle_regressor(batch_emb)
        loss = criterion(pred, batch_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_emb.size(0)
    epoch_loss /= len(dataset)
    print(f"Regressor Epoch {epoch+1}/{REGRESSOR_EPOCHS} - Loss: {epoch_loss:.6f}")

angle_regressor.eval()

label_emb_mats = {}
label_meta = {}
for label, lst in embedding_bank.items():
    embs = np.stack([d['emb'] for d in lst], axis=0)
    label_emb_mats[label] = embs
    label_meta[label] = lst

def predict_from_pil(img_pil, k=5):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb2d = embedding_net(img_tensor).cpu().numpy()[0]

    best_label, best_score = None, -1
    best_meta = None
    for label, emat in label_emb_mats.items():
        sims = cosine_similarity([emb2d], emat)[0]
        top_k_idx = np.argsort(sims)[-k:]
        score = sims[top_k_idx].mean()
        if score > best_score:
            best_score = score
            best_label = label
            best_meta = label_meta[label][top_k_idx[-1]]

    with torch.no_grad():
        emb_t = torch.from_numpy(emb2d.astype(np.float32)).unsqueeze(0).to(device)
        pred_vec = angle_regressor(emb_t).cpu().numpy()[0]
    az = pred_vec[0]
    el = pred_vec[1]
    roll = pred_vec[2]

    return best_label, (az, el, roll), best_score, best_meta

def predict_2d_image(img_path, k=5):
    img = Image.open(img_path).convert("RGB")
    return predict_from_pil(img, k=k)


all_tests, corrects = 0, 0
class_correct = defaultdict(int)
class_total = defaultdict(int)
max_per_class = 300
mistakes = []

for label in labels:
    class_path = os.path.join(test_base_2d, label, "test")
    if not os.path.exists(class_path):
        print(f"No test folder for {label}")
        continue

    image_files = [f for f in os.listdir(class_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files = sorted(image_files)[:max_per_class]

    print(f"\nTesting {label}: {len(image_files)} images")
    for fname in image_files:
        img_path = os.path.join(class_path, fname)
        pred, (az, el, roll), score, meta = predict_2d_image(img_path)

        if pred == label:
            corrects += 1
            class_correct[label] += 1
        else:
            mistakes.append(f"{fname}: predicted={pred}, az={az:.1f}, el={el:.1f}, roll={roll:.1f}, true={label} (score={score:.3f})")
        class_total[label] += 1
        all_tests += 1

        test_img = Image.open(img_path).convert("RGB")

        pred_img_path = meta['img_path'] if meta and os.path.exists(meta['img_path']) else None
        if pred_img_path:
            pred_img = Image.open(pred_img_path).convert("RGB")
        else:
            pred_img = Image.new("RGB", test_img.size, (200, 200, 200))

        plt.figure(figsize=(6, 3))
        plt.suptitle(f"True: {label} | Pred: {pred} | az={az:.1f}, el={el:.1f}, roll={roll:.1f} | score={score:.3f}", fontsize=8)
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title("2D Test Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        plt.title("Predicted Closest Render")
        plt.axis("off")

        compare_path = f"result/visual_compare/{label}_{fname}_compare.jpg"
        plt.savefig(compare_path, bbox_inches="tight")
        plt.close()

if mistakes:
    with open("mistakes1.txt", "w") as f:
        for line in mistakes:
            f.write(line + "\n")

if all_tests > 0:
    print("\n========== RESULTS ==========")
    print(f"General 2D Test Accuracy: {corrects / all_tests:.4f} ({corrects}/{all_tests})")
    print("=============================")
    for label in labels:
        if class_total[label] > 0:
            acc = class_correct[label] / class_total[label]
            print(f"{label:10s}: {acc:.4f} ({class_correct[label]}/{class_total[label]})")
        else:
            print(f"{label:10s}: No test images found")
else:
    print("No test images found")

def resize_keep_aspect(img_cv, height=360):
    """Resize OpenCV (BGR) image to given height keeping aspect ratio."""
    h, w = img_cv.shape[:2]
    scale = height / float(h)
    new_w = int(w * scale)
    return cv2.resize(img_cv, (new_w, height))

def pil_to_cv2(img_pil):
    """PIL RGB -> OpenCV BGR"""
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(frame_bgr):
    """OpenCV BGR -> PIL RGB"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def overlay_text(img, text, origin=(10, 25), scale=0.6, thickness=1):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def run_realtime_camera(camera_idx=0, display_height=360):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Failed to open camera", camera_idx)
        return

    print("Realtime camera mode started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, exiting.")
            break

        pil_img = cv2_to_pil(frame)
        try:
            pred_label, (az, el, roll), score, meta = predict_from_pil(pil_img, k=5)
        except Exception as e:
            print("Prediction error:", e)
            pred_label, az, el, roll, score, meta = "err", 0, 0, 0, 0.0, None

        pred_render_cv = None
        if meta is not None and 'img_path' in meta and os.path.exists(meta['img_path']):
            pred_render_pil = Image.open(meta['img_path']).convert("RGB")
            pred_render_cv = pil_to_cv2(pred_render_pil)
        else:
            ph = display_height
            pw = int(ph * frame.shape[1] / max(frame.shape[0], 1))
            pred_render_cv = np.full((ph, pw, 3), 200, dtype=np.uint8)

        frame_rs = resize_keep_aspect(frame, height=display_height)
        pred_rs = resize_keep_aspect(pred_render_cv, height=display_height)

        info_text = f"Pred: {pred_label} | az={az:.1f} el={el:.1f} roll={roll:.1f} | score={score:.3f}"
        overlay_text(frame_rs, info_text, origin=(10, 20), scale=0.6, thickness=1)

        combined = np.hstack([frame_rs, pred_rs])

        cv2.imshow("Realtime: Camera | Predicted Closest Render", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_camera(camera_idx=0, display_height=360)
