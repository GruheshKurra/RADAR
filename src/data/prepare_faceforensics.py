#!/usr/bin/env python3

import argparse
import cv2
import sys
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging import print_section, print_complete, print_result


MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
MAX_FRAMES_PER_VIDEO = 10
FPS = 1


def extract_frames_from_video(video_path: Path, output_dir: Path, max_frames: int, fps: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    frame_count = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"{video_path.stem}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count


def process_split(ff_dir: Path, split: str, output_dir: Path, max_frames: int, fps: int):
    split_dir = ff_dir / split
    if not split_dir.exists():
        return {"real": 0, "fake": 0}

    real_output = output_dir / split / "real"
    fake_output = output_dir / split / "fake"

    stats = {"real": 0, "fake": 0}

    original_dir = split_dir / "original"
    if original_dir.exists():
        videos = list(original_dir.glob("*.mp4"))
        for video in tqdm(videos, desc=f"  {split}/original"):
            frames = extract_frames_from_video(video, real_output, max_frames, fps)
            stats["real"] += frames

    for manipulation in MANIPULATION_TYPES:
        manip_dir = split_dir / "altered" / manipulation
        if not manip_dir.exists():
            continue

        videos = list(manip_dir.glob("*.mp4"))
        for video in tqdm(videos, desc=f"  {split}/{manipulation}"):
            frames = extract_frames_from_video(video, fake_output, max_frames, fps)
            stats["fake"] += frames

    return stats


def save_metadata(output_dir: Path, stats: dict, config: dict):
    metadata = {
        "dataset": "FaceForensics++",
        "version": "compressed_c23",
        "statistics": stats,
        "config": config
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare FaceForensics++ for training")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to FaceForensics_compressed directory")
    parser.add_argument("--output_dir", type=str, default="./data/faceforensics_frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_VIDEO,
                        help="Maximum frames per video")
    parser.add_argument("--fps", type=int, default=FPS,
                        help="Frames per second to extract")
    args = parser.parse_args()

    video_dir = Path(args.video_dir) / "FaceForensics_compressed"
    output_dir = Path(args.output_dir)

    if not video_dir.exists():
        print(f"✗ Error: {video_dir} not found")
        sys.exit(1)

    print_section(
        "FACEFORENSICS++ FRAME EXTRACTION",
        f"Input: {video_dir}\nOutput: {output_dir}\nMax frames/video: {args.max_frames}\nFPS: {args.fps}"
    )

    all_stats = {}
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")
        stats = process_split(video_dir, split, output_dir, args.max_frames, args.fps)
        all_stats[split] = stats
        print_result(stats, f"  {split}: ")

    config = {"max_frames": args.max_frames, "fps": args.fps}
    save_metadata(output_dir, all_stats, config)

    total_real = sum(s["real"] for s in all_stats.values())
    total_fake = sum(s["fake"] for s in all_stats.values())

    print_complete(
        "FRAME EXTRACTION COMPLETE",
        {
            "Total real frames": total_real,
            "Total fake frames": total_fake,
            "Total frames": total_real + total_fake,
            "Location": str(output_dir)
        }
    )

    print("\nNext step:")
    print(f"  python 2_train_model.py --data_dir {output_dir} --source_domain faceforensics")


if __name__ == "__main__":
    main()
