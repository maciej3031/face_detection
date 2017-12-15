import cv2

from settings import IMG_SIZE
from utils import open_image, convert_to_array, resize_image_to_nxn_square, reshape_to_on_dim_vector


class Canny(object):
    def __init__(self):
        pass

    def get_features(self, img_path):
        img = open_image(img_path)
        img = resize_image_to_nxn_square(img, IMG_SIZE)
        img = cv2.Canny(img, 120, 220)
        array = convert_to_array(img)
        array = reshape_to_on_dim_vector(array)
        return array
