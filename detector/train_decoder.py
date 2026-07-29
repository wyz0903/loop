"""
train_decoder.py — 冻结骨干, 只训练恢复解码器 (Phase 2)
"""
import os, sys, argparse, numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from detector.classifier import Detector
from detector.train import PreprocessedDataset, _latest_data_dir, DEVICE

MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')
BEST_MODEL = os.path.join(MODEL_DIR, 'nn_recovery_best.pt')

BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 80
LAST_STEP_W = 0.3


def main():
    data_dir = _latest_data_dir()
    print(f"Data: {data_dir}")

    norm_data = np.load(os.path.join(data_dir, 'normalizer.npz'))
    norm = {k: norm_data[k] for k in ['ymeas_scale', 'ymeas_median', 'cmd_max']}

    train_ds = PreprocessedDataset(data_dir, split='train', downsample_a0=0.5)
    val_ds = PreprocessedDataset(data_dir, split='val')
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)

    model = Detector(
        ymeas_scale=norm['ymeas_scale'].tolist(),
        ymeas_median=norm['ymeas_median'].tolist(),
        cmd_max=norm['cmd_max'].tolist())
    state = torch.load(BEST_MODEL, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.train()

    # 冻结 backbone + 分类头
    model.kinematic_layer.requires_grad_(False)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.cls_head.requires_grad_(False)
    model.attn_query.requires_grad_(False)
    # 仅解码器可训练
    model.decoder.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_trainable:,} (仅 Decoder)")

    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    best_val = float('inf')
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE, non_blocking=True)
            yc = batch[2].to(DEVICE, non_blocking=True)
            _, _, y_pred = model(x, return_recon=True)
            loss = F.mse_loss(y_pred, yc)
            if LAST_STEP_W > 0:
                loss = (1 - LAST_STEP_W) * loss + LAST_STEP_W * F.mse_loss(y_pred[:, -1, :], yc[:, -1, :])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs
        train_loss = total_loss / n

        # Val
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_last = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE, non_blocking=True)
                yc = batch[2].to(DEVICE, non_blocking=True)
                _, _, y_pred = model(x, return_recon=True)
                loss = F.mse_loss(y_pred, yc)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_n += bs
                val_last += float(torch.sqrt(F.mse_loss(y_pred[:, -1, :], yc[:, -1, :]))) * bs
        val_loss /= val_n
        val_last /= val_n
        scheduler.step(val_loss)

        print(f"E {epoch:3d} | T MSE={train_loss:.4f} | V MSE={val_loss:.4f} Last RMSE={val_last:.4f}", end='')
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL)
            print(" *")
        else:
            print()

    print(f"\nBest epoch {best_epoch}, val MSE={best_val:.4f}")
    model.load_state_dict(torch.load(BEST_MODEL, map_location='cpu', weights_only=True))
    torch.save(model.state_dict(), BEST_MODEL)
    print("Done.")


if __name__ == '__main__':
    main()
