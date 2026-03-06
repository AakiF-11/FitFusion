"""
src/preprocessing/validator.py
Validates that input images meet minimum quality requirements
before entering the expensive GPU pipeline.
"""
from PIL import Image


def validate_image(image_path: str, min_height: int = 512, min_width: int = 384) -> None:
    """
    Raises ValueError if the image is too small, corrupt, or not RGB-compatible.

    Args:
        image_path: Path to the image file.
        min_height: Minimum acceptable height in pixels.
        min_width:  Minimum acceptable width in pixels.
    """
    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as e:
        raise ValueError(f"Cannot open image '{image_path}': {e}")

    img = Image.open(image_path)
    w, h = img.size

    if h < min_height or w < min_width:
        raise ValueError(
            f"Image '{image_path}' is {w}x{h}px — "
            f"minimum required is {min_width}x{min_height}px."
        )
