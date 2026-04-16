import numpy as np
from typing import List, Tuple
import cv2
import os

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    grid = tile_images(images)
    scale_percent = 0.7
    display_grid = cv2.resize(grid, None, fx=scale_percent, fy=scale_percent, interpolation=cv2.INTER_AREA)
    window_title = " | ".join(names)
    cv2.imshow(window_title, display_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for image, filename in zip(images, filenames):
        cv2.imwrite(filename, image)


def scale_down(image: np.array) -> np.array:
    """Returns an image half the size of the original.

    Args:
        image: A numpy array with an opencv image

    Returns:
        A numpy array with an opencv image half the size of the original image
    """
    scale_size = 0.5
    small_img = cv2.resize(image, None , fx = scale_size,fy=scale_size , interpolation=cv2.INTER_AREA)
    return small_img


def separate_channels(colored_image: np.array) -> t_image_triplet:
    """Takes an BGR color image and splits it three images.

    Args:
        colored_image: an numpy array sized [HxWxC] where the channels are in BGR (Blue, Green, Red) order

    Returns:
        A tuple with three BGR images the first one containing only the Blue channel active, the second one only the
        green, and the third one only the red.
    """
    blue_img = colored_image.copy()
    green_img = colored_image.copy()
    red_img = colored_image.copy()
    blue_img[:, :, 1] = 0
    blue_img[:, :, 2] = 0

    green_img[:, :, 0] = 0
    green_img[:, :, 2] = 0

    red_img[:, :, 0] = 0
    red_img[:, :, 1] = 0
    return blue_img, green_img, red_img

def tile_images(image_list, cols=None) -> None:
    """Shows one or more images at once in a tiled manner.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    if not image_list:
        return None
    
    n = len(image_list)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))   
    h,w =  image_list[0].shape[:2]
    blank = np.zeros_like(image_list[0])
    
    all_rows = []
    for r in range(rows):
        row_images = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                row_images.append(image_list[idx])
            else:
                row_images.append(blank)
        all_rows.append(np.hstack(row_images))

    final_grid = np.vstack(all_rows)
    return final_grid

