import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SRDataset
from arch_NAFBL import NAFNetSR


device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= Dataset =================
training = SRDataset(
    r"C:\work\CVPR_2026\wYe7pBJ7-train\train",
    hr_size=(32, 64),
    scale=2,
    augment=True
)

validation = SRDataset(
    r"C:\work\CVPR_2026\wYe7pBJ7-train\valid",
    hr_size=(32, 64),
    scale=2,
    augment=False
)

train_loader = DataLoader(training, batch_size=8, shuffle=True)
val_loader = DataLoader(validation, batch_size=8, shuffle=False)

# ================= Model =================
model = NAFNetSR(
    img_channel=3,
    width=64,
    num_blks=32,
    up_scale=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,   # số epoch full cycle
    eta_min=1e-7          # lr nhỏ nhất
)

best_val_loss = float("inf")

# ================= Training =================
for epoch in range(30):

    # ======== TRAIN ========
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/30 [Train]")

    for lr, hr in pbar:
        lr = lr.to(device)
        hr = hr.to(device)

        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # ======== VALIDATION ========
    
    model.eval()
    val_loss = 0
    val_psnr = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)

            # ----- L1 loss -----
            loss = criterion(sr, hr)
            val_loss += loss.item()

            # ----- PSNR -----
            mse = F.mse_loss(sr, hr, reduction='mean')
            psnr = 10 * torch.log10(1.0 / mse)
            val_psnr += psnr.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)
    scheduler.step()

    print(
        f"\nEpoch {epoch+1}: "
        f"Train Loss = {avg_train_loss:.4f} | "
        f"Val Loss = {avg_val_loss:.4f}"
        f"Val PSNR = {avg_val_psnr:.2f} dB"
    )
    # ======== Save Last ========
    torch.save(model.state_dict(), "last_model.pth")

    # ======== Save Best (based on VAL) ========
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model based on validation loss!\n")
