from skimage.feature import hog
import cv2
import numpy as np

index = np.random.randint(0, 1000)

path1_image = cv2.imread(
    '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/images_with_coloration/0010_0.95_0.16_Equalized.jpg'
)
path2_image = cv2.imread(
    '/mnt/c/Users/Administrator/Desktop/SOTS/outdoor/images_with_coloration/0010_0.95_0.16_finalWithoutCLAHE.jpg'
)
# IMG_DIMS = (128, 64)  # SkIMAGE takes input in HEIGHT X WIDTH format
# image1 = resize(image, IMG_DIMS)
#calculating HOG features
save_path = '/mnt/c/Users/Administrator/Desktop/hog1.jpg'
save_path2 = '/mnt/c/Users/Administrator/Desktop/hog2.jpg'
features, hog_image = hog(path1_image,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          visualize=True,
                          multichannel=True)
cv2.imwrite(save_path, hog_image)
print(features)
features, hog_image = hog(path2_image,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          visualize=True,
                          multichannel=True)
print(features)
cv2.imwrite(save_path2, hog_image)
