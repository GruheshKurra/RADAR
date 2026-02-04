"""RADAR Preprocessing Module.

Design Decision (Research-Critical):
    Frequency features are computed ON-THE-FLY only via `compute_frequency_spectrum`
    in model.py. This ensures bit-identical processing during training and inference,
    eliminating distribution mismatch that could invalidate experimental results.

    Only edge (Sobel) features support optional caching, as their computation is
    deterministic and matches the on-the-fly implementation exactly.

References:
    - compute_frequency_spectrum() in model.py: canonical FFT with high-pass filter
    - BoundaryArtifactDetector.forward() in model.py: on-the-fly Sobel computation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import hashlib


class EdgeExtractor:
    """
    Sobel edge extraction for boundary artifact detection.

    Computes gradient magnitude using Sobel operators, with per-image normalization
    to ensure consistent feature scaling without batch-dependent artifacts.

    The normalization is per-image (not per-batch) to prevent sample interaction
    leakage that could affect model generalization.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Sobel edge features from an image.

        Args:
            image: RGB image array (H, W, 3) or grayscale (H, W)

        Returns:
            Normalized Sobel magnitude map (H, W) in [0, 1]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Per-image normalization (NOT batch-dependent)
        max_val = sobel_magnitude.max()
        if max_val > 0:
            sobel_normalized = (sobel_magnitude / max_val).astype(np.float32)
        else:
            sobel_normalized = sobel_magnitude.astype(np.float32)

        return sobel_normalized


def preprocess_image(img_path: Path, output_dir: Path,
                     edge_extractor: EdgeExtractor) -> bool:
    """
    Preprocess a single image and cache edge features.

    Note: Frequency features are NOT cached. They are computed on-the-fly during
    training/inference to ensure exact consistency with the high-pass filtered
    FFT implementation in model.py.

    Args:
        img_path: Path to input image
        output_dir: Directory to save cached features
        edge_extractor: EdgeExtractor instance for Sobel computation

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(img_path, 'rb') as f:
            img_content = f.read()
        img_hash = hashlib.md5(img_content).hexdigest()

        image = cv2.imdecode(np.frombuffer(img_content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Only cache edge features (frequency computed on-the-fly)
        edge_features = edge_extractor(image_rgb)
        edge_path = output_dir / f"{img_hash}_sobel.npy"
        np.save(str(edge_path), edge_features)

        return True
    except Exception:
        return False


# Backwards compatibility alias (deprecated)
FrequencyExtractor = None  # Removed - use compute_frequency_spectrum() in model.py
