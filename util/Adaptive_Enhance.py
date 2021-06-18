#! /usr/bin/python3
import numpy as np
import math
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter, ImageEnhance
import cv2
import os
import sys

# from util.Adaptive_CLAHE import adaptiveCLAHE
# from util.Adaptive_Gamma_Correction import adaptiveGammaCorrection

# You need to either import the Adaptive-CLAHE / Adap. Gamma scripts first
# or just copy paste the code
#! /usr/bin/python3


# Adaptive CLAHE code:
def CLAHE(channel, limit, tiles):
    """Applies CLAHE via OpenCV implementation (can be used for Saturation or Value Channel of HSV-Space)
    Args:
    channel: 2-D-Numpy-Array (uint8) representing the "Value"/"Saturation"-Channel of the image
    limit: Float, Clip-Limit for CLAHE
    tiles: Tupel, sets the grid size to divide the image into
    Returns:
      Corrected channel 
    """

    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(tiles, tiles))
    return clahe.apply(channel)


def calcEntropy(channel):
    """Calculates the entropy for the current CLAHE-channel
    Args:
    channel: 2-D-Numpy-Array (uint8) representing the "Value"/"Saturation"-Channel of the image
    Returns:
      Entropy 
    """
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]) / channel.size
    entropy = np.sum(hist * np.log2(hist + 1e-7))
    return (-1.0 * entropy)


def calcCurvature(xs, ys):
    """Calculates the curvature of the Clip-Limit vs. Entropy function
    Args:
    xs: List, represents different Clip-Limits
    ys: List, represents corresponding, calculated entropies
    Returns:
      Optimal position (in array) (highest curvature) 
    """
    dx_dt = np.gradient(xs)
    dy_dt = np.gradient(ys)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt +
                                                       dy_dt * dy_dt)**1.5

    #return optimal position
    return np.argmax(curvature)


def adaptiveCLAHE(channel):
    """Wrapper to apply adaptive CLAHE to a channel (H / S-channel of HSV)
  http://onlinepresent.org/proceedings/vol21_2013/52.pdf
    Args:
    channel: 2-D-Numpy-Array (uint8) representing the "Value"/"Saturation"-Channel of the image
    Returns:
      Contrast limited histogram equalized channel ("contrast" only of V-Channel is used ;))
    """

    channel_orig = channel

    #resizing for drastic speedup with minor quality loss
    channel = cv2.resize(channel, (0, 0),
                         fx=0.5,
                         fy=0.5,
                         interpolation=cv2.INTER_AREA)

    basic_limit = 0.5
    expanding = 0.01

    res_entropys = []
    cur_entropy = calcEntropy(channel)
    res_entropys.append(cur_entropy)

    for cnt in range(50):
        tmp_v_CLAHE = CLAHE(channel, basic_limit + expanding * cnt, 8)
        cur_entropy = calcEntropy(tmp_v_CLAHE)
        res_entropys.append(cur_entropy)

    #find and apply optimal cliplimit
    res_entropys = list(map(float, res_entropys))
    opt_Limit = basic_limit + expanding * calcCurvature(
        range(51), res_entropys)

    if opt_Limit < basic_limit:
        opt_Limit = basic_limit

    #adjust window size
    tiles = 6
    res_entropys = []
    tmp_v_CLAHE = CLAHE(channel, opt_Limit, 8)
    cur_entropy = calcEntropy(tmp_v_CLAHE)
    res_entropys.append(cur_entropy)

    for cnt in range(7):
        tmp_v_CLAHE = CLAHE(channel, opt_Limit, tiles + cnt)
        cur_entropy = calcEntropy(tmp_v_CLAHE)
        res_entropys.append(cur_entropy)

    res_entropys = list(map(float, res_entropys))
    opt_tiles = tiles + calcCurvature(range(8), res_entropys)

    #return optimized channel
    return CLAHE(channel_orig, opt_Limit, opt_tiles)


# Adaptive Gamma correction code:
#! /usr/bin/python3


def heaviside(x):
    """Implementation of the Heaviside step function (https://en.wikipedia.org/wiki/Heaviside_step_function)
    Args:
    x: Numpy-Array or single Scalar
    Returns:
    x with step values 	
    """
    if x <= 0:
        return 0
    else:
        return 1


def adaptiveGammaCorrection(v_Channel):
    """https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0138-1#CR14 
	Applies adaptive Gamma-Correction to V-Channels of an HSV-image.

    Args:
    v_Channel: Numpy-Array (uint8) representing the "Value"-Channel of the image
    Returns:
      Corrected channel 
    """

    #calculate general variables
    I_in = v_Channel / 255.0
    I_out = I_in

    sigma = np.std(I_in)
    mean = np.mean(I_in)
    D = 4 * sigma

    #low contrast image
    if D <= 1 / 3:

        gamma = -np.log2(sigma)

        I_in_f = I_in**gamma
        mean_f = (mean**gamma)

        k = I_in_f + (1 - I_in_f) * mean_f

        c = 1 / (1 + heaviside(0.5 - mean) * (k - 1))

        #dark
        if mean < 0.5:
            I_out = I_in_f / ((I_in_f + ((1 - I_in_f) * mean_f)))

        #bright
        else:
            I_out = c * I_in_f

    #high contrast image
    elif D > 1 / 3:

        gamma = np.exp((1 - (mean + sigma)) / 2)

        I_in_f = I_in**gamma
        mean_f = (mean**gamma)

        k = I_in_f + (1 - I_in_f) * mean_f

        c = 1 / (1 + heaviside(0.5 - mean) * (k - 1))

        I_out = c * I_in_f

    else:
        print('Error calculating D')

    I_out = I_out * 255

    return I_out.astype(np.uint8)


# driver code:
# srcDirectory = r'../Originals'
# saveDirectory = '/mnt/c/Users/Administrator/Desktop/image-enhancement'


def adaptive_enhance(img):
    """ Extensive Image-Enhancement featuring Adaptive-CLAHE and Adaptive Gamma-Correction. 
  Colors are additonally improved via a simple gradient guided saturation adjustment.
  Can be used as general script for large amount of images e.g. from vacation, parties, shootings...
  
  Due to the implementation, some parts are unperformant python code. Therefore enhancing a huge amount of high-res images
  will take some time. Better grab a coffee ;)

    Args:
    img: Already read input image
    Returns:
      Returns extensively enhanced Image.
    """

    # filename = os.path.basename(path)
    #uncomment for duplicating the orignal images
    #img = Image.open(path).convert('RGB')
    #img.save(saveDirectory + filename + '.png')

    # img_orig = cv2.imread(path, 1)
    # img = img_orig

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gradient = cv2.Laplacian(s, cv2.CV_32F,
                             ksize=1)  #cv2.Laplacian(s, cv2.CV_32F, ksize = 1)
    clipped_gradient = gradient * np.exp(
        -1 * np.abs(gradient) * np.abs(s - 0.5))

    #normalize to [-1...1]
    clipped_gradient = 2 * (clipped_gradient - np.max(clipped_gradient)
                            ) / -np.ptp(clipped_gradient) - 1
    #clipped_gradient = (clipped_gradient - np.amin(clipped_gradient)) / (np.amax(clipped_gradient) - np.amin(clipped_gradient))
    clipped_gradient = 0.5 * clipped_gradient  #--> 0.5 limits maximum saturation change to 50 %

    factor = np.add(1.0, clipped_gradient)

    s = np.multiply(s, factor)
    s = cv2.convertScaleAbs(s)

    v = adaptiveGammaCorrection(v)
    #v = adaptiveCLAHE(v)
    s = adaptiveCLAHE(s)

    final_CLAHE = cv2.merge((h, s, v))

    #additional sharpening
    tmpimg = cv2.cvtColor(final_CLAHE, cv2.COLOR_HSV2BGR)
    shimg = Image.fromarray(cv2.cvtColor(tmpimg, cv2.COLOR_BGR2RGB))
    sharpener = ImageEnhance.Sharpness(shimg)
    sharpened = sharpener.enhance(2.0)
    # return sharpened

    # converting PIL image to OpenCV format
    op_img = np.array(sharpened)
    op_img = op_img[:, :, ::-1].copy()
    return op_img


# img = cv2.imread(
#     '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/images_with_coloration/0010_0.95_0.16.jpg'
# )
# enh_img = adaptive_enhance(img)
# op_img = np.array(enh_img)
# op_img = op_img[:, :, ::-1].copy()
# print("Enhanced image received.")
# # print(op_img.shape)
# # enh_img.save('/mnt/c/Users/Administrator/Desktop/repl-test/test.jpg')
# cv2.imwrite('/mnt/c/Users/Administrator/Desktop/repl-test/test.jpg', op_img)
# print("Image written")

#Do the magic
# img_path = sys.argv[1]
# gt_path = '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/gt/'
# inp_img = cv2.imread(img_path)
# for filename in os.listdir(srcDirectory):
# 	if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG"):

# enhImg = wrapper(img_path)
# eqlzd_img_path = img_path[:-4] + '_Equalized.jpg'
# equalized_filename = eqlzd_img_path[eqlzd_img_path.rindex('/'):-4]
# clahe_img = cv2.imread(eqlzd_img_path)
# cv2.imwrite(saveDirectory + equalized_filename + '.jpg', clahe_img)
# equalEnhImg = wrapper(eqlzd_img_path)
# filename = img_path[img_path.rindex('/'):-4]
# cv2.imwrite(saveDirectory + filename + '.jpg', inp_img)
# gt_img_name = filename[:filename.find('_')] + '.png'
# gt_img = cv2.imread(gt_path + gt_img_name)
# cv2.imwrite(saveDirectory + gt_img_name, gt_img)
# enhImg.save(saveDirectory + filename + '_enh.jpg', quality=94, optimize=True)
# equalEnhImg.save(saveDirectory + equalized_filename + '_enh.jpg',
#                  quality=94,
#                  optimize=True)
# print(os.path.join(srcDirectory, filename))
