import os
import shutil
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from arch_NAFBL import NAFNetSR


# ==============================
# CONFIG
# ==============================
INPUT_ROOT = r""
OUTPUT_ROOT = r""
MODEL_PATH = "best_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# Load model (GIỐNG TRAIN)
# ==============================
model = NAFNetSR(
    img_channel=3,
    width=64,
    num_blks=32,
    up_scale=2
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = T.ToTensor()
to_pil = T.ToPILImage()

# ==============================
# Copy folder structure
# ==============================
if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)

shutil.copytree(INPUT_ROOT, OUTPUT_ROOT)
print("Folder structure copied.")

# ==============================
# Collect all LR files
# ==============================
lr_files = []

for root, dirs, files in os.walk(OUTPUT_ROOT):
    for file in files:
        if file.startswith("lr-") and file.lower().endswith((".jpg", ".png")):
            lr_files.append(os.path.join(root, file))

print(f"Found {len(lr_files)} LR images")

# ==============================
# Inference
# ==============================
with torch.no_grad():
    for img_path in tqdm(lr_files):

        img = Image.open(img_path).convert("RGB")

        # ⚠️ KHÔNG resize nếu test đã đúng size
        lr = transform(img).unsqueeze(0).to(device)

        sr = model(lr)
        sr = sr.squeeze(0).cpu().clamp(0, 1)

        sr_img = to_pil(sr)

        # Ghi đè lại lr-xxx bằng SR
        sr_img.save(img_path)

print("All LR images replaced with SR output.")
