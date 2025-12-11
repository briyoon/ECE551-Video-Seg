import argparse
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import glob
import subprocess


def create_subset(davis_root, output_root, count=10, seed=42):
    """Create a random subset of the DAVIS dataset."""
    random.seed(seed)

    davis_path = Path(davis_root)
    out_path = Path(output_root)

    # 1. Locate Sequences
    img_root = davis_path / "JPEGImages" / "480p"
    if not img_root.exists():
        img_root = davis_path / "JPEGImages" / "Full-Resolution"

    if not img_root.exists():
        print(f"Error: Could not find JPEGImages in {davis_path}")
        return

    all_seqs = [d.name for d in img_root.iterdir() if d.is_dir()]
    print(f"Found {len(all_seqs)} sequences in {img_root}")

    if len(all_seqs) < count:
        print(f"Warning: Only {len(all_seqs)} available, extracting all.")
        selected = all_seqs
    else:
        selected = random.sample(all_seqs, count)

    print(f"Selected {len(selected)}: {selected}")

    # 2. Copy Data
    for seq in tqdm(selected, desc="Copying sequences"):
        # Source Paths
        src_img = davis_path / "JPEGImages" / "480p" / seq
        src_ann = davis_path / "Annotations" / "480p" / seq

        # Fallback to Full-Res if 480p missing (though DAVIS Usually has both or one)
        if not src_img.exists():
            src_img = davis_path / "JPEGImages" / "Full-Resolution" / seq
        if not src_ann.exists():
            src_ann = davis_path / "Annotations" / "Full-Resolution" / seq

        # Dest Paths
        dst_img = out_path / "JPEGImages" / "480p" / seq
        dst_ann = out_path / "Annotations" / "480p" / seq

        # Copy Images
        if src_img.exists():
            shutil.copytree(src_img, dst_img, dirs_exist_ok=True)

        # Copy Annotations (if exist)
        if src_ann.exists():
            shutil.copytree(src_ann, dst_ann, dirs_exist_ok=True)

    # 3. Create ImageSet file
    imageset_dir = out_path / "ImageSets" / "2017"
    imageset_dir.mkdir(parents=True, exist_ok=True)

    with open(imageset_dir / "train.txt", "w") as f:
        for seq in selected:
            f.write(f"{seq}\n")

    print(f"Successfully created subset in {output_root}")


def generate_videos_from_subset(davis_root_or_subset, output_dir=None):
    """Generate MP4 videos (Source & GT) from a DAVIS-structured folder."""
    root = Path(davis_root_or_subset)

    # Read sequences from ImageSets if exists, else scan folders
    set_file = root / "ImageSets" / "2017" / "train.txt"
    if set_file.exists():
        with open(set_file) as f:
            seqs = [line.strip() for line in f if line.strip()]
    else:
        # Scan dir
        img_root = root / "JPEGImages" / "480p"
        if not img_root.exists():
            print(f"Could not find valid image directory in {root}")
            return
        seqs = [d.name for d in img_root.iterdir() if d.is_dir()]

    # Output dir defaults to 'Videos' inside the subset, or specified path
    if output_dir:
        vis_dir = Path(output_dir)
    else:
        vis_dir = root / "Videos"

    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating videos for {len(seqs)} sequences into {vis_dir}...")

    for seq in tqdm(seqs, desc="Rendering MP4s"):
        img_dir = root / "JPEGImages" / "480p" / seq
        ann_dir = root / "Annotations" / "480p" / seq

        # 1. Source Video
        if img_dir.exists():
            out_vid = vis_dir / f"{seq}.mp4"
            if not out_vid.exists():
                images_to_video(img_dir, out_vid, is_annotation=False)

        # 2. GT Video
        if ann_dir.exists():
            out_gt = vis_dir / f"{seq}_gt.mp4"
            if not out_gt.exists():
                images_to_video(ann_dir, out_gt, is_annotation=True)

    print("Video generation complete.")


def images_to_video(img_dir, output_path, fps=24, is_annotation=False):
    """Convert directory of images to video."""
    ext = "*.png" if is_annotation else "*.jpg"
    images = sorted(glob.glob(str(img_dir / ext)))

    if not images:
        return

    frame0 = cv2.imread(images[0])
    h, w, _ = frame0.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Simple palette for GT (BGR)
    palette = {
        0: (0, 0, 0),
        1: (0, 0, 255),  # Red
        2: (0, 255, 0),  # Green
        3: (255, 0, 0),  # Blue
        4: (0, 255, 255),  # Yellow
        5: (255, 0, 255),  # Magenta
        6: (255, 255, 0),  # Cyan
    }

    for img_path in images:
        if is_annotation:
            pil_img = Image.open(img_path)
            mask = np.array(pil_img)

            color_img = np.zeros((h, w, 3), dtype=np.uint8)
            for oid in np.unique(mask):
                if oid == 0:
                    continue
                bgr = palette.get(oid, (255, 255, 255))
                color_img[mask == oid] = bgr

            video.write(color_img)
        else:
            frame = cv2.imread(img_path)
            video.write(frame)

    video.release()

    # Convert to H.264 using ffmpeg for browser compatibility
    try:
        temp_path = str(output_path) + ".temp.mp4"
        os.rename(str(output_path), temp_path)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",  # Critical for browser support
                "-crf",
                "23",
                "-preset",
                "fast",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        os.remove(temp_path)
    except Exception as e:
        print(
            f"Warning: FFMPEG conversion failed for {output_path}, keeping original. Error: {e}"
        )
        if os.path.exists(temp_path):
            os.rename(temp_path, str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAVIS Subset & Video Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: create
    cmd_create = subparsers.add_parser("create", help="Create a random subset")
    cmd_create.add_argument(
        "--davis", required=True, help="Path to original DAVIS dataset"
    )
    cmd_create.add_argument(
        "--output", required=True, help="Path to output subset folder"
    )
    cmd_create.add_argument("--count", type=int, default=10, help="Number of sequences")
    cmd_create.add_argument("--seed", type=int, default=42, help="Random seed")

    # Command: videos
    cmd_vid = subparsers.add_parser("videos", help="Generate MP4s from a subset")
    cmd_vid.add_argument(
        "--subset", required=True, help="Path to the DAVIS subset folder"
    )
    cmd_vid.add_argument(
        "--output", help="Optional output folder for videos (default: subset/Videos)"
    )

    args = parser.parse_args()

    if args.command == "create":
        create_subset(args.davis, args.output, args.count, args.seed)
        # Optional: Auto-generate videos after creation?
        # generate_videos_from_subset(args.output)
    elif args.command == "videos":
        generate_videos_from_subset(args.subset, args.output)
