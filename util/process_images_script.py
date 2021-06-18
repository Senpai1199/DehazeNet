import os
import cv2
from Adaptive_Enhance import adaptive_enhance
from CLAHE import apply_CLAHE

with_coloration_path = '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/testForImageCharacteristics/with-coloration/'
without_coloration_path = '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/testForImageCharacteristics/without-coloration/'

enhanced_image_save_path = '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/AdaptiveClahe+Gamma1/'

for filename in os.listdir(with_coloration_path):
    im_path = with_coloration_path + filename
    inp_img = cv2.imread(im_path)
    op_img = adaptive_enhance(inp_img)
    cv2.imwrite(enhanced_image_save_path + filename, op_img)
    print("Processed image: ", filename)

for filename in os.listdir(without_coloration_path):
    im_path = without_coloration_path + filename
    inp_img = cv2.imread(im_path)
    op_img = adaptive_enhance(inp_img)
    cv2.imwrite(enhanced_image_save_path + filename, op_img)
    print("Processed image: ", filename)
