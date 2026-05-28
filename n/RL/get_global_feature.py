import os
import sys
import numpy as np
import numpy.linalg as LA
import time
import cv2
import scipy.ndimage
from skimage import color
import matplotlib.image as mpimg
import multiprocessing as mtp
import scipy.interpolate


def get_tone_spatial_distribution(L, num_interval):
    edges = np.linspace(0.0, 100.0, num_interval + 1)
    ftr = np.zeros((num_interval, 3), dtype=np.single)
    for i in range(num_interval):
        idx = np.nonzero((L >= edges[i]) & (L < edges[i + 1]))
        num_pix = len(idx[0])
        if num_pix > 0:
            cy = np.mean(idx[0])
            cx = np.mean(idx[1])
            std_dev_y = np.sqrt(np.sum((idx[0] - cy) ** 2) / num_pix)
            std_dev_x = np.sqrt(np.sum((idx[1] - cx) ** 2) / num_pix)
            ftr[i, 0] = std_dev_x * std_dev_y / num_pix
            ftr[i, 1:3] = [cx, cy]
        else:
            ftr[i, :] = 0
    return ftr.reshape((num_interval * 3))


def get_highlight_clipping_value(L, percentage):
    h, w = L.shape[0], L.shape[1]
    rank = int(np.round(h * w * percentage))  # ensure integer index
    sorted_L = np.sort(L.flatten())
    return sorted_L[-rank]


def get_BSpline_curve(x, y, num_control_points, t_min, t_max):
    p = 3
    n = num_control_points - 1
    m = p + n + 1
    t = np.linspace(t_min, t_max, m - p - p + 1)
    t = t[1:-1]
    spline = scipy.interpolate.LSQUnivariateSpline(x, y, t)
    return spline


def get_cum_hist_BSpline_curve(cumsum_hist, t_min, t_max, bins):
    p = 3
    n = 50
    m = p + n + 1
    step = (t_max - t_min) / bins
    t = np.linspace(t_min, t_max, m - p - p + 1)
    assert len(t) == (m - p - p + 1)
    t = t[1:-1]
    x = np.linspace(t_min + step * 0.5, t_max - step * 0.5, bins)
    spline = scipy.interpolate.LSQUnivariateSpline(x, cumsum_hist, t)
    return spline, x


def get_lightness_equalization_curve_control_points(L):
    h, w = L.shape[0], L.shape[1]
    bins = 100
    hist, bin_edges = np.histogram(L, bins, range=(0, 100), density=False)
    hist = np.single(hist) / np.single(h * w)
    cumsum_hist = np.cumsum(hist)
    spline, x = get_cum_hist_BSpline_curve(cumsum_hist, 0.0, 100.0, bins)
    return spline.get_coeffs()


def get_lightness_detail_weighted_equalization_curve_control_points(L, sigma):
    bins = 100
    grad_mag = scipy.ndimage.gaussian_gradient_magnitude(L, sigma)
    hist, bin_edges = np.histogram(L, bins, range=(0, 100), weights=grad_mag)
    hist = np.single(hist) / np.sum(grad_mag)
    cumsum_hist = np.cumsum(hist)
    spline, x = get_cum_hist_BSpline_curve(cumsum_hist, 0.0, 100.0, bins)
    return spline.get_coeffs()


def get_img_lightness_hist(L, range_min=0, range_max=100, bins=50):
    h, w = L.shape[0], L.shape[1]
    L1 = scipy.ndimage.gaussian_filter(L, sigma=10, order=0)
    L2 = scipy.ndimage.gaussian_filter(L, sigma=20, order=0)

    # 'normed' was removed in NumPy 1.24+; use density=False (manual normalisation below)
    hist, _ = np.histogram(L.flatten(), bins, range=(range_min, range_max), density=False)
    hist = np.single(hist) / np.single(h * w)
    hist1, _ = np.histogram(L1.flatten(), bins, range=(range_min, range_max), density=False)
    hist1 = np.single(hist1) / np.single(h * w)
    hist2, _ = np.histogram(L2.flatten(), bins, range=(range_min, range_max), density=False)
    hist2 = np.single(hist2) / np.single(h * w)
    return np.concatenate((hist, hist1, hist2))


def get_global_feature(np_img, log_luminance=1):
    Lab_img = color.rgb2lab(np_img + 0.5)
    L = Lab_img[:, :, 0]
    h, w = L.shape[0], L.shape[1]

    if log_luminance == 1:
        idx = np.nonzero(L == 0)
        L_copy = np.zeros((h, w), dtype=np.single)
        L_copy[:, :] = L[:, :]
        L_copy[idx[0], idx[1]] = 1
        log_L = np.log(L_copy)

    if log_luminance == 1:
        lightness_hist = get_img_lightness_hist(log_L, 0, np.log(100))
    else:
        lightness_hist = get_img_lightness_hist(L, 0, 100)

    cp1 = get_lightness_equalization_curve_control_points(L)
    cp2 = get_lightness_detail_weighted_equalization_curve_control_points(L, 1)
    cp3 = get_lightness_detail_weighted_equalization_curve_control_points(L, 10)
    cp4 = get_lightness_detail_weighted_equalization_curve_control_points(L, 20)

    hl_clipping = np.zeros((6), dtype=np.single)
    hl_clipping[0] = get_highlight_clipping_value(L, 0.01)
    hl_clipping[1] = get_highlight_clipping_value(L, 0.02)
    hl_clipping[2] = get_highlight_clipping_value(L, 0.03)
    hl_clipping[3] = get_highlight_clipping_value(L, 0.05)
    hl_clipping[4] = get_highlight_clipping_value(L, 0.1)
    hl_clipping[5] = get_highlight_clipping_value(L, 0.15)

    L_spatial_distr = get_tone_spatial_distribution(L, 10)

    return np.concatenate((lightness_hist, cp1, cp2, cp3, cp4, hl_clipping, L_spatial_distr))
