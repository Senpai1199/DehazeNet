import cv2
import numpy as np


def apply_CLAHE(image):
    """
        Receives image read using opencv and returns the contrast enhanced image
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # r_image, g_image, b_image = cv2.split(image)

    # r_image_eq = cv2.equalizeHist(r_image)
    # g_image_eq = cv2.equalizeHist(g_image)
    # b_image_eq = cv2.equalizeHist(b_image)

    # image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
    # return image_eq
    bgr = image
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr
    # cv2.imwrite('/mnt/c/Users/Administrator/Desktop/img-output.jpg', bgr)


# image = cv2.imread('/mnt/c/Users/Administrator/Desktop/test.jpg')
# apply_CLAHE(image)
