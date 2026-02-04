#!/usr/bin/env python3

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List
import multiprocessing as mp
from functools import partial


def extract_frames_from_video(video_path: Path, output_dir: Path,
                               num_frames: int = 10, quality: int = 95) -> int:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return 0

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        video_name = video_path.stem
        frames_to_write = []

        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = output_dir / f"{video_name}_frame{idx:03d}.jpg"
                frames_to_write.append((str(frame_path), frame, quality))

        cap.release()

        extracted = 0
        for frame_path, frame, qual in frames_to_write:
            if cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, qual]):
                extracted += 1

        return extracted

    except:
        return 0


def process_video_wrapper(args):
    video_path, output_dir, num_frames, quality = args
    return extract_frames_from_video(video_path, output_dir, num_frames, quality)


def extract_ff_dataset(ff_root: Path, output_root: Path,
                      num_frames: int = 10, num_workers: int = None):
    fake_folders = [
        "DeepFakeDetection",
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures"
    ]
    real_folder = "original"

    if num_workers is None:
        num_workers = max(1, mp.cpu_count())

    output_dataset = output_root / "ff_c23"
    real_dir = output_dataset / "real"
    fake_dir = output_dataset / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting FaceForensics++ frames...")
    print(f"Input: {ff_root}")
    print(f"Output: {output_dataset}")
    print(f"Frames per video: {num_frames}")
    print(f"Workers: {num_workers}")

    print("\n[1/2] Processing REAL videos (1000 expected)...")
    real_path = ff_root / real_folder
    real_videos = []
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
    print(f"Found {len(real_videos)} real videos")

    if real_videos:
        args_list = [(v, real_dir, num_frames, 95) for v in real_videos]
        chunk_size = max(1, len(args_list) // (num_workers * 4))

        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_video_wrapper, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Real videos"
            ))

        total_real_frames = sum(results)
        print(f"Extracted {total_real_frames} real frames")
    else:
        total_real_frames = 0
        print("No real videos found")

    print("\n[2/2] Processing FAKE videos (6000 expected)...")
    fake_videos = []
    for folder in fake_folders:
        folder_path = ff_root / folder
        if folder_path.exists():
            videos = list(folder_path.glob("*.mp4"))
            fake_videos.extend(videos)
            print(f"  {folder}: {len(videos)} videos")

    print(f"Total fake videos: {len(fake_videos)}")

    if fake_videos:
        args_list = [(v, fake_dir, num_frames, 95) for v in fake_videos]
        chunk_size = max(1, len(args_list) // (num_workers * 4))

        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_video_wrapper, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Fake videos"
            ))

        total_fake_frames = sum(results)
        print(f"Extracted {total_fake_frames} fake frames")
    else:
        total_fake_frames = 0
        print("No fake videos found")

    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print(f"Real videos processed: {len(real_videos)}/1000 expected")
    print(f"Fake videos processed: {len(fake_videos)}/6000 expected")
    print(f"Real frames: {total_real_frames}")
    print(f"Fake frames: {total_fake_frames}")
    print(f"Total frames: {total_real_frames + total_fake_frames}")
    print(f"Output directory: {output_dataset}")
    print(f"\nDataset structure:")
    print(f"  {output_dataset}/real/  ({len(list(real_dir.glob('*.jpg')))} images)")
    print(f"  {output_dataset}/fake/  ({len(list(fake_dir.glob('*.jpg')))} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from FaceForensics++ videos"
    )
    parser.add_argument(
        "--ff_root",
        type=str,
        required=True,
        help="Path to FaceForensics++ root directory (contains original/, Deepfakes/, etc.)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data",
        help="Output root directory (default: ./data)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames to extract per video (default: 10)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    args = parser.parse_args()

    ff_root = Path(args.ff_root)
    output_root = Path(args.output_root)

    if not ff_root.exists():
        print(f"Error: FaceForensics++ directory not found: {ff_root}")
        return

    extract_ff_dataset(ff_root, output_root, args.num_frames, args.num_workers)


if __name__ == "__main__":
    main()
