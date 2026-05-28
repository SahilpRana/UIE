"""
Guided Filter implementation.
Reference: K. He, J. Sun, and X. Tang, "Guided Image Filtering", ECCV 2010.
"""

import numpy as np
import cv2


class GuidedFilter:
    """
    Guided Filter for edge-preserving smoothing / transmission map refinement.

    Parameters
    ----------
    guide : np.ndarray
        Guide image (uint8 H×W×3 or H×W).  Used to compute local statistics.
    radius : int
        Filter radius (window size = 2*radius + 1).
    eps : float
        Regularisation parameter that controls smoothing strength.
    """

    def __init__(self, guide, radius: int, eps: float):
        if guide.dtype != np.float64:
            guide = guide.astype(np.float64)

        # Convert colour guide to greyscale for a single-channel guided filter
        if guide.ndim == 3:
            self._guide = (
                0.2126 * guide[:, :, 0]
                + 0.7152 * guide[:, :, 1]
                + 0.0722 * guide[:, :, 2]
            ) / 255.0
        else:
            self._guide = guide / 255.0 if guide.max() > 1.0 else guide

        self._r = radius
        self._eps = eps
        self._wsize = 2 * radius + 1


    def _box(self, arr: np.ndarray) -> np.ndarray:
        """Per-pixel box (mean) filter with the stored window size."""
        return cv2.boxFilter(
            arr, ddepth=-1,
            ksize=(self._wsize, self._wsize),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )


    def filter(self, src: np.ndarray) -> np.ndarray:
        """
        Apply the guided filter to *src* using the guide image stored at
        construction time.

        Parameters
        ----------
        src : np.ndarray
            Input / source image (float, same spatial size as the guide).

        Returns
        -------
        np.ndarray
            Filtered output, same shape and dtype as *src*.
        """
        src_f = src.astype(np.float64)
        I = self._guide  # noqa: E741  (I is the conventional variable name)

        mean_I = self._box(I)
        mean_p = self._box(src_f)
        mean_Ip = self._box(I * src_f)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self._box(I * I)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self._eps)
        b = mean_p - a * mean_I

        mean_a = self._box(a)
        mean_b = self._box(b)

        output = mean_a * I + mean_b
        return output.astype(src.dtype)
