import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import hashlib


class FrequencyExtractor:

    Note: This preprocessing version supports both DCT and FFT for flexibility.
    For inference, FrequencyArtifactDetector in model.py uses the canonical FFT
    implementation (see compute_frequency_spectrum). The preprocessing cache is
    optional and only provides a speedup; models work correctly without it.

    def __init__(self, use_dct: bool = False):
        self.use_dct = use_dct
        if use_dct:
            import torch_dct
            self.torch_dct = torch_dct

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray_normalized = gray.astype(np.float32) / 255.0

        if self.use_dct:
            gray_tensor = torch.from_numpy(gray_normalized).unsqueeze(0).unsqueeze(0)
            dct = self.torch_dct.dct_2d(gray_tensor)
            freq_magnitude = torch.abs(dct).squeeze().numpy()
        else:
            fft = np.fft.fft2(gray_normalized)
            fft_shifted = np.fft.fftshift(fft)
            freq_magnitude = np.abs(fft_shifted)

        freq_magnitude = np.log1p(freq_magnitude)
        freq_magnitude = (freq_magnitude - freq_magnitude.min()) / (freq_magnitude.max() - freq_magnitude.min() + 1e-8)

        return freq_magnitude.astype(np.float32)


class EdgeExtractor:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        max_val = sobel_magnitude.max()
        if max_val > 0:
            sobel_normalized = (sobel_magnitude / max_val).astype(np.float32)
        else:
            sobel_normalized = sobel_magnitude.astype(np.float32)

        return sobel_normalized


def preprocess_image(img_path: Path, output_dir: Path,
                     freq_extractor: FrequencyExtractor,
                     edge_extractor: EdgeExtractor) -> bool:
    try:
        with open(img_path, 'rb') as f:
            img_content = f.read()
        img_hash = hashlib.md5(img_content).hexdigest()

        image = cv2.imdecode(np.frombuffer(img_content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        freq_features = freq_extractor(image_rgb)
        edge_features = edge_extractor(image_rgb)

        freq_path = output_dir / f"{img_hash}_freq.npy"
        edge_path = output_dir / f"{img_hash}_sobel.npy"

        np.save(str(freq_path), freq_features)
        np.save(str(edge_path), edge_features)

        return True
    except Exception:
        return False
