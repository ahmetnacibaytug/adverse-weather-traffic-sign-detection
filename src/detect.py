import argparse
import csv
from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--data", type=Path, required=True, help="YOLO dataset root")
    p.add_argument("--yaml", default="datamix.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", type=Path, default=Path("runs/detection"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--ignore-speed-limit-10", action="store_true")
    a = p.parse_args()
    root = a.data.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((root / a.yaml).read_text())
    cfg["path"] = str(root)
    model = YOLO(a.weights)
    if a.mode == "train":
        path = a.out / "train.yaml"
        path.write_text(yaml.safe_dump(cfg))
        model.train(data=str(path.resolve()), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch_size, project=str(a.out.resolve()), name="train")
        return
    names = cfg["names"]
    names = names.items() if isinstance(names, dict) else enumerate(names)
    keep = [int(i) for i,n in names if str(n).replace("_", " ").lower() != "speed limit 10"] if a.ignore_speed_limit_10 else None
    rows = []
    for split in ("test", "snow_test", "rainy_test", "test2"):
        images = root / split / "images"
        if not images.is_dir(): raise FileNotFoundError(images)
        scenario = dict(cfg, test=str(images))
        path = a.out / f"{split}.yaml"
        path.write_text(yaml.safe_dump(scenario))
        result = model.val(data=str(path.resolve()), split="test", imgsz=a.imgsz, conf=0.001, iou=0.5, classes=keep, plots=True, project=str(a.out.resolve()), name=split)
        p_, r = float(result.box.mp), float(result.box.mr)
        rows.append([split, p_, r, 2*p_*r/(p_+r) if p_+r else 0, float(result.box.map50), float(result.box.map)])
    with (a.out / "detection_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["split","precision","recall","f1","map50","map50_95"]); w.writerows(rows)

if __name__ == "__main__": main()
