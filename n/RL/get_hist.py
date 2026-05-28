import numpy as np
import skimage.color as color
from PIL import Image
import cv2
from sklearn.preprocessing import normalize


# All image input is RGB whitened to (-0.5, 0.5).
# Returns the flattened histogram.

def rgbl_hist(image_data):
    r_h, _ = np.histogram(image_data[:, :, 0], 256, [-0.5, 0.5])
    g_h, _ = np.histogram(image_data[:, :, 1], 256, [-0.5, 0.5])
    b_h, _ = np.histogram(image_data[:, :, 2], 256, [-0.5, 0.5])
    l = (image_data[:, :, 0] * 0.2126
         + image_data[:, :, 1] * 0.7152
         + image_data[:, :, 2] * 0.0722)
    l_h, _ = np.histogram(l, 256, [-0.5, 0.5])
    return np.concatenate((r_h, g_h, b_h, l_h), axis=0) / 1000.0


def tiny_hist_old(image_np):
    image_np_rgb = image_np + 0.5  # 0~1 rgb
    image_np_lab = color.rgb2lab(image_np_rgb)
    image_np_lab_tiny = cv2.resize(image_np_lab, (16, 16), interpolation=cv2.INTER_CUBIC)
    return image_np_lab_tiny.reshape(16 * 16 * 3) / 100.0


def tiny_hist(image_np):
    image_pil = Image.fromarray(np.uint8((image_np + 0.5) * 255))
    image_pil.thumbnail((16, 16), Image.LANCZOS)
    image_np_resized = np.asarray(image_pil) / 255.0
    image_np_lab = color.rgb2lab(image_np_resized)
    return image_np_lab.reshape(16 * 16 * 3) / 100.0


def tiny_hist_28(image_np):
    image_pil = Image.fromarray(np.uint8((image_np + 0.5) * 255))
    image_pil.thumbnail((28, 28), Image.LANCZOS)
    image_np_resized = np.asarray(image_pil) / 255.0
    image_np_lab = color.rgb2lab(image_np_resized)
    return image_np_lab.reshape(28 * 28 * 3) / 100.0


def lab_hist(image_np):
    image_np_rgb = image_np + 0.5  # 0~1 rgb
    image_np_lab = color.rgb2lab(image_np_rgb)

    num_bin_L, num_bin_a, num_bin_b = 10, 10, 10
    image_np_lab = image_np_lab.reshape([224 * 224, 3])
    H, _ = np.histogramdd(
        image_np_lab,
        bins=(num_bin_L, num_bin_a, num_bin_b),
        range=((0, 100), (-60, 60), (-60, 60)),
    )
    return H.reshape(1000) / 1000.0


def lab_hist_8k(image_np):
    image_np_rgb = image_np + 0.5  # 0~1 rgb
    image_np_lab = color.rgb2lab(image_np_rgb)

    num_bin_L, num_bin_a, num_bin_b = 20, 20, 20
    image_np_lab = image_np_lab.reshape([224 * 224, 3])
    H, _ = np.histogramdd(
        image_np_lab,
        bins=(num_bin_L, num_bin_a, num_bin_b),
        range=((0, 100), (-60, 60), (-60, 60)),
    )
    return normalize(H.reshape(1, 8000)).ravel()
