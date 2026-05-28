import numpy as np
from skimage import exposure
import cv2


from albumentations import Sharpen, Emboss

from dcpmain import getRecoverScene


def contrast(image_rgb, b):
    mean = np.mean(image_rgb)
    degenerate = np.zeros(image_rgb.shape) + mean
    image_rgb = b * image_rgb + (1 - b) * degenerate
    image_rgb = np.clip(image_rgb, 0, 1)
    return image_rgb


def brightness(image_rgb, b):
    degenerate = np.zeros(image_rgb.shape)
    image_rgb = b * image_rgb + (1 - b) * degenerate
    image_rgb = np.clip(image_rgb, 0, 1)
    return image_rgb


def color_saturation(image_rgb, b):
    degenerate = image_rgb.mean(axis=2)
    image_rgb[:, :, 0] = b * image_rgb[:, :, 0] + (1 - b) * degenerate
    image_rgb[:, :, 1] = b * image_rgb[:, :, 1] + (1 - b) * degenerate
    image_rgb[:, :, 2] = b * image_rgb[:, :, 2] + (1 - b) * degenerate
    image_rgb = np.clip(image_rgb, 0, 1)
    return image_rgb


def white_bal(image_rgb, r, g, b):
    image_rgb[:, :, 0] = r / 255.0 * image_rgb[:, :, 0]
    image_rgb[:, :, 1] = g / 255.0 * image_rgb[:, :, 1]
    image_rgb[:, :, 2] = b / 255.0 * image_rgb[:, :, 2]
    image_rgb = np.clip(image_rgb, 0, 1)
    return image_rgb


def Gamma_up(image_rgb):
    image_rgb[:, :, 0] = exposure.adjust_gamma(image_rgb[:, :, 0], gamma=0.7)
    image_rgb[:, :, 1] = exposure.adjust_gamma(image_rgb[:, :, 1], gamma=0.7)
    image_rgb[:, :, 2] = exposure.adjust_gamma(image_rgb[:, :, 2], gamma=0.7)
    return image_rgb


def Gamma_down(image_rgb):
    image_rgb[:, :, 0] = exposure.adjust_gamma(image_rgb[:, :, 0], gamma=1.3)
    image_rgb[:, :, 1] = exposure.adjust_gamma(image_rgb[:, :, 1], gamma=1.3)
    image_rgb[:, :, 2] = exposure.adjust_gamma(image_rgb[:, :, 2], gamma=1.3)
    return image_rgb


def HE(image_rgb):
    image = np.uint8(image_rgb * 255)
    for i in range(3):
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return image / 255


def CLAHE(image_rgb):
    image = np.uint8(image_rgb * 255)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        image[:, :, i] = clahe.apply(image[:, :, i])
    return image / 255


def white_balance(image_rgb, percent=0):
    image = np.uint8(image_rgb * 255)
    out_channels = []
    cumstops = (
        image.shape[0] * image.shape[1] * percent / 200.0,
        image.shape[0] * image.shape[1] * (1 - percent / 200.0),
    )
    for channel in cv2.split(image):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut),
        ))
        out_channels.append(cv2.LUT(channel, lut.astype("uint8")))
    image = cv2.merge(out_channels)
    return image / 255


def sharpen(image_rgb):
    image = np.uint8(image_rgb * 255)
    # Sharpen replaces the removed IAASharpen (albumentations >= 1.0)
    aug = Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1)
    result = aug(image=image)["image"]
    return result / 255


def emboss(image_rgb):
    image = np.uint8(image_rgb * 255)
    # Emboss replaces the removed IAAEmboss (albumentations >= 1.0)
    aug = Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=1)
    result = aug(image=image)["image"]
    return result / 255


def DCP(image_rgb):
    image = np.uint8(image_rgb * 255)
    _transmission, sceneRadiance = getRecoverScene(image)
    return sceneRadiance / 255


def DCP_simple(image_rgb):
    image = image_rgb.copy()
    darkImage = image_rgb.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - 0.6 * (darkImage / maxDarkChannel)
    t[t < 0.1] = 0.1

    J = image_rgb.copy()
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    return J
