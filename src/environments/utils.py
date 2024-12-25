import numpy as np
import cv2
from PIL import Image


def match_tp(img, template, thres=0.8, overlap_thres=0.1, x_range=[48, 200]):
    img = img[x_range[0] : x_range[1]]

    h, w = template.shape[:2]

    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    overlap_thres_h = max(int(h * overlap_thres), 1)
    overlap_thres_w = max(int(w * overlap_thres), 1)

    # (#match, 2)
    loc = np.vstack(np.where(result >= thres)).T

    if len(loc) == 0:
        return []

    loc = np.vstack(sorted(loc, key=lambda l: result[l[0]][l[1]], reverse=True))

    # move the points from the top left to the center of the shape
    loc += [h // 2, w // 2]

    filterd_loc = []

    for p1 in loc:
        overlapped = False
        for p2 in filterd_loc:
            # check overlap
            diff_h, diff_w = np.abs(p1 - p2)

            if diff_h < h - overlap_thres_h and diff_w < w - overlap_thres_w:
                overlapped = True

        if not overlapped:
            filterd_loc.append(p1)

    # then sort by position
    filterd_loc = sorted(filterd_loc, key=lambda x: (x[0], x[1]))

    return np.vstack(filterd_loc) + [x_range[0], 0]


def load_img_np(img_path):
    pic = Image.open(img_path)
    return np.array(pic)


def rgb_to_grayscale(image):
    # Check if the image has 3 channels
    if image.shape[-1] == 3:
        # Use the formula to convert to grayscale
        gray_image = (
            0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
        )
        return gray_image.astype(np.uint8)  # Convert to uint8 type
    else:
        raise ValueError("Input image must be a 3-channel RGB image.")
