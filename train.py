import argparse, json
import torchvision, numpy as np
from pathlib import Path
from datetime import datetime
from fastai.vision.all import *
from fastai.callback.all import SaveModelCallback, ReduceLROnPlateau, EarlyStoppingCallback, CSVLogger, GradientClip
from torchvision.models import detection

# ─── Module‐level state & helpers (so they can be pickled) ──────────────────

keypoints_data = []  # loaded in main()

def get_keypoints(img_path):
    "Lookup keypoints by image stem."
    stem = Path(img_path).stem
    for e in keypoints_data:
        if Path(e['image']).stem == stem:
            return [[float(x), float(y)] for x,y in e['keypoints']]
    return [[0.0,0.0],[0.0,0.0]]

def keypoint_distance(preds, targs):
    "Mean Euclidean distance over our two keypoints."
    p = preds.view(-1,2,2)
    return ((p - targs)**2).sum(dim=-1).sqrt().mean()

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train Kururu keypoint model with full logging & callbacks")
    p.add_argument('--backbone',        type=str,   default='resnet34')
    p.add_argument('--lowres',          type=int,   default=224)
    p.add_argument('--highres',         type=int,   default=448)
    p.add_argument('--lowbs',           type=int,   default=32)
    p.add_argument('--highbs',          type=int,   default=16)
    p.add_argument('--epochs_low',      type=int,   default=30)
    p.add_argument('--epochs_high',     type=int,   default=20)
    p.add_argument('--lr',              type=float, default=2e-3)
    p.add_argument('--fine_tune_lr',    type=float, default=1e-4)
    p.add_argument('--fine_tune_epochs',type=int,   default=20)
    p.add_argument('--early_stop',      action='store_true')
    p.add_argument('--continue_from',   type=str,   help="Path to .pkl to resume from")
    args = p.parse_args()

    # ─ Paths ──────────────────────────────────────────────────────────────────
    labels_folder = Path('data/train/labels')
    images_folder = Path('data/train/images')
    models_dir    = Path('models'); models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir      = Path('data/logs');   logs_dir.mkdir(parents=True, exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d")

    # ─ Load annotations ────────────────────────────────────────────────────────
    global keypoints_data
    jsons = sorted(labels_folder.glob('data_*.json'))
    if not jsons: raise FileNotFoundError(f"No JSON in {labels_folder}")
    with open(jsons[-1]) as f: keypoints_data = json.load(f)

    # ─ DataBlock & DataLoaders ────────────────────────────────────────────────
    dblock = DataBlock(
        blocks=(ImageBlock, PointBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(0.2, seed=42),
        get_y=get_keypoints,
        item_tfms=Resize(args.lowres),
        batch_tfms=Normalize.from_stats(*imagenet_stats)
    )
    dls = dblock.dataloaders(images_folder, bs=args.lowbs)

    # ─ Learner & Callbacks ────────────────────────────────────────────────────
    arch = getattr(torchvision.models, args.backbone)
    
    if arch == "resnet34":
        arch = resnet34(weights=ResNet34_Weights.DEFAULT)
    if arch == "resnet50":
        arch = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    learn = vision_learner(
        dls, arch,
        loss_func=keypoint_distance,
        y_range=(-1,1),
        metrics=[mae]
    ).to_fp16()

    # resume if requested
    if args.continue_from:
        print(f"⚡ Resuming from {args.continue_from}")
        learn = load_learner(args.continue_from)

    # 1) Save best model whenever mae improves by ≥ 1e-5
    save_cb = SaveModelCallback(
        monitor='mae',
        fname=f"best_{args.backbone}",
        comp=np.less,
        min_delta=1e-5
    )
    # 2) Log every epoch to CSV
    csv_cb = CSVLogger(
        fname=logs_dir/f"train-{args.backbone}-{dt}.csv",
        append=False
    )
    # 3) LR scheduler
    lr_cb = ReduceLROnPlateau(monitor='mae',
                              factor=0.1,
                              patience=10,
                              min_delta=1e-4)

    cbs = [
        save_cb,
        csv_cb,
        lr_cb,
        GradientClip(max_norm=0.1)
    ]

    # optional early stopping
    if args.early_stop:
        cbs.append(EarlyStoppingCallback(
            monitor='mae',
            patience=max(1, args.epochs_low//5)
        ))

    # ─ Phase 1: Low‐res training ───────────────────────────────────────────────
    print(f"▶ Phase 1: {args.lowres}px @ BS={args.lowbs}, epochs={args.epochs_low}")
    learn.fine_tune(args.epochs_low, base_lr=args.lr, cbs=cbs)

    # ─ Phase 2: Progressive resize ─────────────────────────────────────────────
    print(f"▶ Phase 2: {args.highres}px @ BS={args.highbs}, epochs={args.epochs_high}")
    dls = dblock.new(item_tfms=Resize(args.highres))\
               .dataloaders(images_folder, bs=args.highbs)
    learn.dls = dls
    learn.fine_tune(args.epochs_high, base_lr=args.lr, cbs=cbs)

    # ─ Phase 3: Fine‐tuning ────────────────────────────────────────────────────
    print(f"▶ Phase 3: Fine‐tune @ LR={args.fine_tune_lr}, epochs={args.fine_tune_epochs}")
    learn.fine_tune(args.fine_tune_epochs, base_lr=args.fine_tune_lr, cbs=cbs)

    # ─ Export the best model ─────────────────────────────────────────────────────────
    learn.load(f"best_{args.backbone}")
    out = models_dir/f"kururu-{args.backbone}-{dt}.pkl"
    learn.export(out)
    print(f"✅ Training complete! Best model → {out}")

if __name__=="__main__":
    main()

