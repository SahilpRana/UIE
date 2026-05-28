from PIL import Image, ImageEnhance
import skimage.color as color
import numpy as np
import time
import math
import random
from action_set import *
#from action_set_tf import *


action_size = 20

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)




def take_action(image_np, action_idx):
    # image_pil = Image.fromarray(np.uint8(image_np))
    # image_pil = Image.fromarray(np.uint8((image_np+0.5)*255))
    # enhance contrast
    return_np = None

    if action_idx == 0:
        return_np = Gamma_up(image_np + 0.5)
    elif action_idx == 1:
        return_np = Gamma_down(image_np + 0.5)
    elif action_idx == 2:
        return_np = contrast(image_np+0.5, 0.95)
    elif action_idx == 3:
        return_np = contrast(image_np+0.5, 1.05)
    # enhance color
    elif action_idx == 4:
        return_np = color_saturation(image_np+0.5, 0.95)
    elif action_idx == 5:
        return_np = color_saturation(image_np+0.5, 1.05)
    # color brightness
    elif action_idx == 6:
        return_np = brightness(image_np+0.5, 0.95)
    elif action_idx == 7:
        return_np = brightness(image_np+0.5, 1.05)
    elif action_idx == 8:
        r, g, b = 245, 255, 255  # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 9:
        r, g, b = 265, 255, 255  # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 10:
        r, g, b = 255, 245, 255  # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 11:
        r, g, b = 255, 265, 255  # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 12:
        r, g, b = 255, 255, 245  # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 13:
        r, g, b = 255, 255, 265  # around 6300K #convert_K_to_RGB(6000)
        return_np = white_bal(image_np + 0.5, r, g, b)
    elif action_idx == 14:
        return_np = HE(image_np + 0.5)
    elif action_idx == 15:
        return_np = CLAHE(image_np + 0.5)
    elif action_idx == 16:
        return_np = white_balance(image_np + 0.5, 0.5)
    elif action_idx == 17:
        return_np = sharpen(image_np + 0.5)
    elif action_idx == 18:
        return_np = emboss(image_np + 0.5)
    elif action_idx == 19:
            return_np = DCP(image_np + 0.5)
    else:
      print("error")
    return return_np - 0.5
 