'''
    This script serves as a check for which path the input image should go along to get the
    best haze free and visually appropriate image possible for a given input image.
    There are two paths that an input image can take:
    1. Input image -> Contrast enhancement using CLAHE 
    2. Input image -> DehazeNet (without any contrast enhancement)
    This script returns '1' or '2' accordingly based on some checks done on the input image it receives.
'''

import numpy as np


def get_path(input_image, threshold=142):
    im_array = np.array(input_image)

    red_chan = np.ndarray.flatten(im_array[:, :, 0])
    green_chan = np.ndarray.flatten(im_array[:, :, 1])
    blue_chan = np.ndarray.flatten(im_array[:, :, 2])

    mean_red = np.mean(red_chan)
    mean_green = np.mean(green_chan)
    mean_blue = np.mean(blue_chan)

    red_range = np.amax(red_chan)
    green_range = np.amax(green_chan)
    blue_range = np.amax(blue_chan)

    pixel_threshold = threshold  # threshold pixel brightness value (trial and error)
    pixel_range_threshold = 98

    red_c1 = mean_red > pixel_threshold
    green_c1 = mean_green > pixel_threshold
    blue_c1 = mean_blue > pixel_threshold

    conditions_count = 0
    if red_c1:
        conditions_count += 1
    if green_c1:
        conditions_count += 1
    if blue_c1:
        conditions_count += 1

    red_c2 = red_range > pixel_range_threshold
    green_c2 = green_range > pixel_range_threshold
    blue_c2 = blue_range > pixel_range_threshold

    if conditions_count >= 2:  # return path=1 if 2 or more conditions are True
        path = 1  # just CLAHE (Image will probably show coloration if passed through Dehazenet)
    else:
        path = 2  # just Dehazenet (Image will probably not show coloration if passed through Dehazenet)

    return path