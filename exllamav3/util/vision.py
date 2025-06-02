import numpy as np
from PIL import Image
import math

def convert_to_rgb(image: Image) -> Image:
    """
    Converts an image to RGB format, ensuring any transparent regions are converted to white
    """
    if image.mode == "RGB":
        return image

    image = image.convert("RGBA")

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("RGB")
    return new_image


def normalize_image(
    image: np.ndarray,
    mean: tuple,
    std: tuple,
) -> np.ndarray:
    """
    Normalizes RGB image in numpy format using the mean and standard deviation specified by `mean` and `std`:
    image = (image - mean(image)) / std
    """

    assert len(mean) == 3 and len(std) == 3, \
        "mean and std arguments must be 3D"

    # Upcast image to float32 if it's not already a float type

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)
    image = (image - mean) / std
    return image
