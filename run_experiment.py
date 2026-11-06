import base64
import io
import json
import os
import platform
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from rembg import new_session, remove
from skimage import data
import torch
import torchvision
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
IMAGES = {}
for index in range(1, 4):
    encoded = (ROOT / "input" / f"painting{index}.webp.b64").read_text(encoding="ascii")
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    IMAGES[f"P{index}"] = image
IMAGES["N1"] = Image.fromarray(data.chelsea()).convert("RGB")
IMAGES["N2"] = Image.fromarray(data.coffee()).convert("RGB")

def mask_record(a):
    packed = np.packbits(a.astype(np.uint8).ravel()).tobytes()
    return {"width": int(a.shape[1]), "height": int(a.shape[0]),
            "packbits_zlib_base64": base64.b64encode(zlib.compress(packed, 9)).decode("ascii")}

def panel_triptych(image, m1, m2, label1, label2, target):
    maxside = 500
    base = image.copy()
    base.thumbnail((maxside, maxside), Image.Resampling.LANCZOS)
    size = base.size
    parts = [base]
    for a in (m1, m2):
        small = Image.fromarray((a * 255).astype("uint8")).resize(size, Image.Resampling.NEAREST)
        tint = Image.new("RGB", size, (240, 65, 75))
        over = Image.blend(base, tint, 0.45)
        parts.append(Image.composite(over, base, small))
    w, h = size
    canvas = Image.new("RGB", (w * 3, h + 35), "white")
    draw = ImageDraw.Draw(canvas)
    for i, part in enumerate(parts):
        canvas.paste(part, (w * i, 35))
        draw.text((w * i + 6, 10), ("Original", label1, label2)[i], fill="black")
    canvas.save(target, quality=82)

def segment():
    print("Loading U2-Net and U2-NetP from rembg GitHub release", flush=True)
    sessions = {k: new_session(k, providers=["CPUExecutionProvider"]) for k in ("u2net", "u2netp")}
    output = {}
    for name in ("P1", "P2", "P3"):
        image = IMAGES[name]
        masks = {}
        for model, session in sessions.items():
            gray = remove(image, session=session, only_mask=True)
            if isinstance(gray, bytes):
                gray = Image.open(io.BytesIO(gray))
            arr = np.asarray(gray.convert("L").resize(image.size, Image.Resampling.LANCZOS))
            masks[model] = arr >= 128
        u, v = masks["u2net"], masks["u2netp"]
        union = np.logical_or(u, v).sum()
        output[name] = {
            "image_size": [image.width, image.height],
            "target": "foreground salient object",
            "threshold": 128,
            "u2net": {"pixels": int(u.sum()), "area_ratio": float(100 * u.mean()),
                       "mask": mask_record(u)},
            "u2netp": {"pixels": int(v.sum()), "area_ratio": float(100 * v.mean()),
                        "mask": mask_record(v)},
            "iou": float(np.logical_and(u, v).sum() / union) if union else 1.0
        }
        panel_triptych(image, u, v, "U2-Net", "U2-NetP", OUT / f"{name}_segmentation.jpg")
        print(name, "U2-Net", round(100*u.mean(), 3), "U2-NetP", round(100*v.mean(), 3), flush=True)
    return output

def classify():
    weights = {"resnet18": ResNet18_Weights.IMAGENET1K_V1,
               "mobilenet_v3_small": MobileNet_V3_Small_Weights.IMAGENET1K_V1}
    networks = {
        "resnet18": resnet18(weights=weights["resnet18"]).eval(),
        "mobilenet_v3_small": mobilenet_v3_small(weights=weights["mobilenet_v3_small"]).eval()
    }
    cats = weights["resnet18"].meta["categories"]
    assert cats == weights["mobilenet_v3_small"].meta["categories"]
    output = {}
    for name in ("N1", "N2", "P1"):
        img = IMAGES[name]
        items = {}
        for model, net in networks.items():
            inp = weights[model].transforms()(img).unsqueeze(0)
            with torch.inference_mode():
                probs = net(inp).softmax(dim=1)[0]
            scores, indices = probs.topk(3)
            items[model] = [{"index": int(k), "label": cats[int(k)], "score": float(s)}
                            for s, k in zip(scores, indices)]
        output[name] = items
        img.thumbnail((420, 420), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max(900, img.width), img.height+155), "white")
        canvas.paste(img, ((canvas.width-img.width)//2, 0))
        draw = ImageDraw.Draw(canvas)
        for i, (model, preds) in enumerate(items.items()):
            x = 12 if i == 0 else canvas.width // 2 + 12
            draw.text((x, img.height+8), model, fill="black")
            for j, v in enumerate(preds):
                draw.text((x, img.height+28+j*25), f"{j+1}. {v['label']}: {v['score']:.4f}", fill="black")
        canvas.save(OUT / f"{name}_classification.jpg", quality=85)
        print(name, [(k, v[0]["label"]) for k, v in items.items()], flush=True)
    return output

if __name__ == "__main__":
    for k, v in IMAGES.items():
        v.save(OUT / f"{k}_input.png")
    result = {
        "order": ["mmexport1789698947182.jpg",
                  "mmexport1789698949163.jpg",
                  "mmexport1789698951426.jpg"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__,
                        "device": "GitHub Actions ubuntu-latest CPU"},
        "segmentation": segment(),
        "classification": classify(),
        "natural_source": {
            "N1": "scikit-image data.chelsea (photograph of a cat)",
            "N2": "scikit-image data.coffee (photograph of a cup of coffee)"
        }
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print("EXPERIMENT_COMPLETE", flush=True)
