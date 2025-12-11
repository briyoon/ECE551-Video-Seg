import argparse
import json
import glob
from pathlib import Path
import numpy as np
import zipfile
import cv2  # used for resizing predictions if needed
from pycocotools import mask as mask_utils
from PIL import Image


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0


def evaluate(davis_root, export_zip):
    """Evaluate exported JSON against DAVIS 2017 Ground Truth."""
    print(f"Evaluating {export_zip} against {davis_root}...")

    # 1. Unzip export
    with zipfile.ZipFile(export_zip, "r") as zf:
        if "train.json" not in zf.namelist():
            print("Error: train.json not found in the zip export.")
            return
        with zf.open("train.json") as f:
            predictions = json.load(f)

    # 2. Map Prediction Video IDs to Names
    vid_id_to_name = {}
    for vid in predictions["videos"]:
        name = Path(vid["name"]).stem
        vid_id_to_name[vid["id"]] = name

    # 3. Group predictions by video and category
    preds_map = {}

    for ann in predictions["annotations"]:
        vid_id = ann["video_id"]
        vid_name = vid_id_to_name.get(vid_id)
        if not vid_name:
            continue

        if vid_name not in preds_map:
            preds_map[vid_name] = []

        # Decode all RLEs for this track
        track_masks = []
        for seg in ann["segmentations"]:
            if seg:
                track_masks.append(mask_utils.decode(seg))
            else:
                track_masks.append(None)

        preds_map[vid_name].append(track_masks)

    # 4. Compute Metrics
    ious = []

    # Iterate over Ground Truth videos
    gt_root = Path(davis_root) / "Annotations" / "480p"
    if not gt_root.exists():
        print("GT annotations not found (checked Annotations/480p)")
        return

    # Evaluating only on videos present in prediction
    for vid_name, track_preds in preds_map.items():
        gt_dir = gt_root / vid_name
        if not gt_dir.exists():
            print(f"Warning: No GT for {vid_name}")
            continue

        gt_files = sorted(glob.glob(str(gt_dir / "*.png")))
        if not gt_files:
            continue

        # Load GT masks
        gt_frames = []
        for f in gt_files:
            img = np.array(Image.open(f))
            gt_frames.append(img)

        # Identify unique object IDs in GT
        obj_ids = np.unique(np.concatenate(gt_frames))
        obj_ids = obj_ids[obj_ids != 0]

        vid_ious = []

        for oid in obj_ids:
            # Construct 3D GT volume
            gt_vol = np.stack([(frame == oid).astype(np.uint8) for frame in gt_frames])
            best_iou = 0.0

            for pred_track in track_preds:
                # Align lengths
                L = min(len(gt_vol), len(pred_track))
                if L == 0:
                    continue

                frame_ious = []
                for i in range(L):
                    gt_m = gt_vol[i]
                    pred_m = pred_track[i]

                    if pred_m is None:
                        frame_ious.append(0.0 if gt_m.sum() > 0 else 1.0)
                    else:
                        if pred_m.shape != gt_m.shape:
                            pred_m = cv2.resize(
                                pred_m,
                                (gt_m.shape[1], gt_m.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        frame_ious.append(compute_iou(pred_m, gt_m))

                if frame_ious:
                    mean_iou = np.mean(frame_ious)
                    if mean_iou > best_iou:
                        best_iou = mean_iou

            vid_ious.append(best_iou)
            print(f"  {vid_name} obj {oid}: IoU = {best_iou:.3f}")

        ious.extend(vid_ious)

    print("-" * 30)
    if ious:
        print(f"Mean IoU (J-score approximation): {np.mean(ious):.4f}")
    else:
        print("No IoUs computed. Check if video names match.")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAVIS Evaluation Tool")

    # Single command mode implies evaluate, but let's be explicit or just default
    parser.add_argument("--davis", required=True, help="Path to DAVIS root (or subset)")
    parser.add_argument("--export", required=True, help="Path to exported project zip")

    args = parser.parse_args()

    evaluate(args.davis, args.export)
