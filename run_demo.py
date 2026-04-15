from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from degradation_functions import apply_blur, apply_jpeg_artifacts, apply_random_brightness_contrast
from text_degradations import create_text_mask, apply_pepper_noise_to_text
from advanced_text_damage import apply_advanced_text_damage


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def center_crop_or_resize(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if min(height, width) < size:
        scale = size / min(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
        height, width = image.shape[:2]
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return image[top : top + size, left : left + size]


def fuse_foreground_with_background(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    foreground_float = foreground.astype(np.float32) / 255.0
    background_float = background.astype(np.float32) / 255.0
    fused = foreground_float * background_float
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def build_demo_variants(clean_image: np.ndarray, background_patch: np.ndarray) -> dict[str, np.ndarray]:
    fused = fuse_foreground_with_background(clean_image, background_patch)
    text_mask = create_text_mask(clean_image)

    variants: dict[str, np.ndarray] = {
        "fused": fused,
        "blur": apply_blur(fused.copy()),
        "jpeg_artifacts": apply_jpeg_artifacts(fused.copy()),
        "brightness_contrast": apply_random_brightness_contrast(fused.copy()),
        "pepper_text_dropout": apply_pepper_noise_to_text(
            fused.copy(),
            text_mask,
            background_patch,
            amount=0.08,
        ),
        "advanced_text_damage": apply_advanced_text_damage(
            fused.copy(),
            text_mask,
            background_patch,
            damage_recipe=random.choice(["classic", "compact_blot", "aggressive_spread"]),
            damage_intensity=0.25,
        ),
    }
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a few Manchu degradation examples from one clean text image.")
    parser.add_argument("--clean", type=Path, required=True, help="Path to a clean text image.")
    parser.add_argument("--background", type=Path, required=True, help="Path to a paper/background texture image.")
    parser.add_argument("--size", type=int, default=512, help="Output crop size.")
    parser.add_argument("--output", type=Path, default=ROOT / "demo_output", help="Directory for generated images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_image = center_crop_or_resize(load_bgr_image(args.clean), args.size)
    background_patch = center_crop_or_resize(load_bgr_image(args.background), args.size)

    variants = build_demo_variants(clean_image, background_patch)
    cv2.imwrite(str(output_dir / "clean.png"), clean_image)
    cv2.imwrite(str(output_dir / "background.png"), background_patch)

    for name, image in variants.items():
        cv2.imwrite(str(output_dir / f"{name}.png"), image)

    print(f"Saved {len(variants)} degraded variants to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

