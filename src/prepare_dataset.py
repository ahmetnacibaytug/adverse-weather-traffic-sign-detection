import argparse
from pathlib import Path
from PIL import Image
import yaml

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--yaml", default="data.yaml")
    p.add_argument("--splits", nargs="+", default=["train_mix", "valid", "test", "snow_test", "rainy_test", "test2"])
    a = p.parse_args()
    names = yaml.safe_load((a.data / a.yaml).read_text())["names"]
    for split in a.splits:
        image_dir = a.data / split / "images"
        if not image_dir.is_dir(): raise FileNotFoundError(image_dir)
        dest_split = "val" if split == "valid" else split
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}: continue
            label = a.data / split / "labels" / (image_path.stem + ".txt")
            if not label.is_file(): continue
            with Image.open(image_path) as raw:
                im = raw.convert("RGB"); w, h = im.size
                for i, line in enumerate(label.read_text().splitlines()):
                    fields = line.split()
                    if len(fields) != 5: continue
                    c, x, y, bw, bh = map(float, fields); c = int(c)
                    name = names.get(c, names.get(str(c))) if isinstance(names, dict) else names[c] if 0 <= c < len(names) else None
                    if name is None: continue
                    name = str(name).replace("_", " ")
                    if name == "Speed Limit 10": continue
                    if Path(name).name != name or name in {".", ".."}: raise ValueError("Invalid label name")
                    box = (max(0,int((x-bw*0.575)*w)), max(0,int((y-bh*0.575)*h)), min(w,int((x+bw*0.575)*w)), min(h,int((y+bh*0.575)*h)))
                    if box[2]-box[0]<16 or box[3]-box[1]<16: continue
                    dest = a.out / dest_split / name; dest.mkdir(parents=True, exist_ok=True)
                    im.crop(box).save(dest / f"{image_path.stem}_crop{i}.jpg")

if __name__ == "__main__": main()
