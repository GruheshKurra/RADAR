import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
        A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform,
        preprocess_dir: Optional[Path] = None,
        validate_cache: bool = True,
    ):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.preprocess_dir = Path(preprocess_dir) if preprocess_dir else None

        if self.preprocess_dir and validate_cache:
            self._validate_preprocessed_features()

    def _validate_preprocessed_features(self):
        missing_count = 0
        total_count = len(self.image_paths)

        for idx, img_path in enumerate(self.image_paths):
            freq_path = self._get_cache_path(img_path, "freq")
            sobel_path = self._get_cache_path(img_path, "sobel")

            if freq_path is None or sobel_path is None or not (freq_path.exists() and sobel_path.exists()):
                missing_count += 1

        if missing_count > 0:
            logger.warning(f"Missing preprocessed features for {missing_count}/{total_count} images")

    def _get_cache_path(self, img_path: Path, feature_type: str) -> Optional[Path]:
        if self.preprocess_dir is None:
            return None

        import hashlib
        try:
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
        except:
            img_hash = hashlib.md5(str(img_path).encode()).hexdigest()

        class_name = "real" if "/real/" in str(img_path) or "\\real\\" in str(img_path) else "fake"

        domain_name = "unknown"
        for domain in ["stylegan", "cifake", "wilddeepfake", "ff_c23"]:
            if domain in str(img_path):
                domain_name = domain
                break

        cache_dir = self.preprocess_dir / domain_name / class_name
        return cache_dir / f"{img_hash}_{feature_type}.npy"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, int, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, int]
    ]:
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            img_array = np.array(img)
        except Exception as e:
            logger.error(f"Failed to load image {self.image_paths[idx]}: {e}")
            if idx + 1 < len(self.image_paths):
                return self.__getitem__(idx + 1)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            img = self.transform(image=img_array)["image"] if self.transform else torch.zeros(3, 224, 224)
            return img, self.labels[idx]

        if self.transform is not None:
            img = self.transform(image=img_array)["image"]

        extras = {}
        if self.preprocess_dir:
            freq_path = self._get_cache_path(self.image_paths[idx], "freq")
            sobel_path = self._get_cache_path(self.image_paths[idx], "sobel")

            if freq_path is not None and freq_path.exists():
                extras["freq_cached"] = torch.from_numpy(np.load(str(freq_path)))

            if sobel_path is not None and sobel_path.exists():
                extras["sobel_cached"] = torch.from_numpy(np.load(str(sobel_path)))

        if extras:
            return img, self.labels[idx], extras
        else:
            return img, self.labels[idx]
