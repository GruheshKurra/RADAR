import numpy as np
from pathlib import Path
from typing import List, Tuple


def load_domain_data(data_dir: Path, domain: str) -> Tuple[List[Path], List[int]]:
    domain_path = data_dir / domain

    if not domain_path.exists():
        raise ValueError(f"Domain directory does not exist: {domain_path}")

    images, labels = [], []

    for class_name, label in [("real", 0), ("fake", 1)]:
        class_dir = domain_path / class_name
        if not class_dir.exists():
            continue

        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in class_dir.glob(ext):
                images.append(img_path)
                labels.append(label)

    if len(images) == 0:
        raise ValueError(f"No images found in {domain_path}")

    real_count = labels.count(0)
    fake_count = labels.count(1)
    min_required = 100
    if real_count < min_required or fake_count < min_required:
        raise ValueError(
            f"Insufficient samples in {domain}: real={real_count}, fake={fake_count}. "
            f"Need at least {min_required} of each class. "
            f"Check 1_prepare_dataset.py - the dataset may not have been downloaded correctly."
        )

    return images, labels


def create_stratified_split(
    images: List[Path],
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
    rng = np.random.RandomState(seed)

    real_idx = [i for i, l in enumerate(labels) if l == 0]
    fake_idx = [i for i, l in enumerate(labels) if l == 1]

    rng.shuffle(real_idx)
    rng.shuffle(fake_idx)

    def split_class(indices):
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = indices[:n_train]
        val = indices[n_train:n_train + n_val]
        test = indices[n_train + n_val:]
        return train, val, test

    real_train, real_val, real_test = split_class(real_idx)
    fake_train, fake_val, fake_test = split_class(fake_idx)

    train_idx = real_train + fake_train
    val_idx = real_val + fake_val
    test_idx = real_test + fake_test

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_images = [images[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_images = [images[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels
