"""Train the YOLO lesion detector. Run `python -m mammo.prepare_yolo` first."""
import argparse
import shutil

from ultralytics import YOLO

from mammo.config import MODELS, YOLO_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="0 for CUDA, mps, cpu")
    p.add_argument("--fraction", type=float, default=1.0, help="use part of train set (smoke tests)")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(YOLO_DIR / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        project=str(MODELS / "runs"),
        name="detector",
        exist_ok=True,
        patience=12,
        # Mammography-friendly augmentation: no mosaic/mixup, no vertical flips, no hue/saturation.
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        scale=0.2,
        translate=0.05,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
    )
    best = MODELS / "runs" / "detector" / "weights" / "best.pt"
    shutil.copy(best, MODELS / "detector.pt")
    print(f"saved {MODELS / 'detector.pt'}")


if __name__ == "__main__":
    main()
