"""
Generate a balanced 50-cat / 50-dog subset of Oxford-IIIT-Pet
and evaluate COCO-style predictions against its ground truth.

Folder layout after `generate` (default --outdir=data/oxford_pet_subset):

outdir/
└── subset/
    ├── images/            ← JPEGs
    ├── masks/             ← binary PNGs (white = pet, black = background)
    └── ground_truth.json  ← COCO annotations
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm


def prepare_dataset(dest: Path) -> Path:
    ds = OxfordIIITPet(
        dest, split="trainval", target_types="segmentation", download=True
    )
    return Path(ds._base_folder)


def _sample_even(pool: Dict[str, List[str]], total: int) -> List[str]:
    breeds = list(pool)
    random.shuffle(breeds)
    base, extra = divmod(total, len(breeds))

    out: List[str] = []
    for i, br in enumerate(breeds):
        take = base + (1 if i < extra else 0)
        if len(pool[br]) < take:
            raise RuntimeError(f"{br}: have {len(pool[br])}, need {take}")
        out.extend(random.sample(pool[br], take))
    return out


def build_subset(
    root: Path, n_per_species: int = 50, seed: int = 42
) -> List[Tuple[str, str, str]]:
    random.seed(seed)

    imgs_dir = root / "images"
    masks_dir = root / "annotations" / "trimaps"
    list_txt = root / "annotations" / "list.txt"

    cat_pool, dog_pool = defaultdict(list), defaultdict(list)

    with list_txt.open() as f:
        lines = f.readlines()[6:]

    for line in lines:
        if not line.strip():
            continue
        name, _cls, species_id, _bid = line.split()
        species_id = int(species_id)
        breed = "_".join(Path(name).stem.split("_")[:-1])
        (cat_pool if species_id == 1 else dog_pool)[breed].append(Path(name).stem)

    cats = _sample_even(cat_pool, n_per_species)
    dogs = _sample_even(dog_pool, n_per_species)

    subset = [
        (str(imgs_dir / f"{n}.jpg"), str(masks_dir / f"{n}.png"), "cat") for n in cats
    ]
    subset += [
        (str(imgs_dir / f"{n}.jpg"), str(masks_dir / f"{n}.png"), "dog") for n in dogs
    ]
    return subset


def load_masks_by_filename(coco_path: Path) -> Dict[str, Dict]:
    from pycocotools.coco import COCO

    coco = COCO(str(coco_path))
    masks: Dict[str, Dict] = {}

    for img in coco.imgs.values():
        ann_ids = coco.getAnnIds(imgIds=[img["id"]])
        if not ann_ids:
            continue

        merged = np.zeros((img["height"], img["width"]), dtype=bool)
        for aid in ann_ids:
            merged |= coco.annToMask(coco.loadAnns([aid])[0]).astype(bool)

        key = img["file_name"]
        masks[key] = {
            "mask": merged,
            "cat": coco.loadAnns([ann_ids[0]])[0]["category_id"],
        }

    return masks


def encode(mask: np.ndarray) -> Dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    return rle


def subset_to_coco(subset: List[Tuple[str, str, str]], out_json: Path) -> None:
    images, anns = [], []
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]

    ann_id = 1
    for img_id, (img_p, msk_p, species) in enumerate(subset, 1):
        w, h = Image.open(img_p).size
        images.append(
            {"id": img_id, "file_name": Path(img_p).name, "width": w, "height": h}
        )

        m = np.array(Image.open(msk_p), dtype=np.uint8)
        bin_mask = m == 1
        rle = encode(bin_mask)
        anns.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1 if species == "cat" else 2,
                "segmentation": rle,
                "area": int(bin_mask.sum()),
                "bbox": list(mask_utils.toBbox(rle)),
                "iscrowd": 0,
            }
        )
        ann_id += 1

    out_json.write_text(
        json.dumps(
            {"images": images, "annotations": anns, "categories": categories}, indent=2
        )
    )
    print(f"✅  Ground truth written to {out_json}")


def load_masks(coco_path: Path) -> Dict[int, Dict]:
    from pycocotools.coco import COCO

    coco = COCO(str(coco_path))

    masks = {}
    for img in coco.imgs.values():
        ann_ids = coco.getAnnIds(imgIds=[img["id"]])
        if not ann_ids:
            continue
        merged = np.zeros((img["height"], img["width"]), dtype=bool)
        for aid in ann_ids:
            merged |= coco.annToMask(coco.loadAnns([aid])[0]).astype(bool)
        masks[img["id"]] = {
            "mask": merged,
            "cat": coco.loadAnns([ann_ids[0]])[0]["category_id"],
        }
    return masks


def iou_dice(gt: np.ndarray, pr: np.ndarray) -> Tuple[float, float]:
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return float("nan"), float("nan")
    return inter / union, 2 * inter / (gt.sum() + pr.sum())


def evaluate(pred_json: Path, gt_json: Path) -> None:
    gt = load_masks_by_filename(gt_json)
    pr = load_masks_by_filename(pred_json)

    ious, dices, per_cls = [], [], defaultdict(list)
    missed = 0

    for fname, gdat in gt.items():
        if fname not in pr:
            missed += 1
            continue

        gt_mask = gdat["mask"]
        pr_mask = pr[fname]["mask"]

        if gt_mask.shape != pr_mask.shape:
            raise ValueError(
                f"Shape mismatch for {fname}: GT {gt_mask.shape} vs PR {pr_mask.shape}"
            )

        i, d = iou_dice(gt_mask, pr_mask)
        ious.append(i)
        dices.append(d)
        per_cls[gdat["cat"]].append(i)

    print(f"Images evaluated    : {len(ious)}")
    print(f"Missing predictions : {missed}")
    print(f"Mean IoU            : {np.nanmean(ious):.4f}")
    print(f"Mean Dice           : {np.nanmean(dices):.4f}")
    for cid, vals in per_cls.items():
        lbl = "cat" if cid == 1 else "dog"
        print(f"IoU ({lbl})            : {np.nanmean(vals):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="download & create balanced subset")
    g.add_argument("--outdir", type=Path, default=Path("data/oxford_pet_subset"))
    g.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("evaluate", help="evaluate prediction JSON")
    e.add_argument("pred_json", type=Path)
    e.add_argument(
        "--gt_json",
        type=Path,
        help="ground-truth JSON (defaults to subset/ground_truth.json)",
    )

    args = ap.parse_args()

    if args.cmd == "generate":
        root = prepare_dataset(args.outdir)
        subset = build_subset(root, seed=args.seed)

        dst_img = args.outdir / "subset" / "images"
        dst_msk = args.outdir / "subset" / "masks"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_msk.mkdir(parents=True, exist_ok=True)

        for im_p, msk_p, _ in tqdm(subset, desc="Copying"):
            shutil.copy(im_p, dst_img / Path(im_p).name)
            m = np.array(Image.open(msk_p), dtype=np.uint8)
            bin_vis = (m == 1).astype(np.uint8) * 255
            Image.fromarray(bin_vis, mode="L").save(dst_msk / Path(msk_p).name)

        subset_to_coco(subset, args.outdir / "subset" / "ground_truth.json")

    elif args.cmd == "evaluate":
        gt_json = args.gt_json or args.pred_json.parent / "ground_truth.json"
        evaluate(args.pred_json, gt_json)


if __name__ == "__main__":
    main()
