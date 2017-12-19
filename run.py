import argparse
import pickle

import sklearn
from canny import Canny
from settings import IMG_SIZE
from utils import jj, open_image, resize_image_to_nxn_square, reshape_to_on_dim_vector, convert_to_array
import cv2
# import matplotlib.pyplot as plt

# TODO: Add some graphical interface
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', help='Input file name. Default = data.txt', default=jj('data', 'face1.jpg'))
    args = p.parse_args()

    # Open image and resize it
    img = open_image(args.input)
    img = resize_image_to_nxn_square(img, IMG_SIZE)

    # Canny features extraction
    features_extractor = Canny(img, 60, 120)
    feature_matrix = features_extractor.get_features()

    # im = cv2.Canny(img, 120, 220)
    # feature_matrix = convert_to_array(img)
    # plt.imshow(im)
    # plt.show()

    # Reshape to one dim
    one_dim_feature_vector = reshape_to_on_dim_vector(feature_matrix)

    # TODO: Implement SVM learning, get rid of pickle
    # Perform classification using SVM or Logistic Regression
    SVM_model = pickle.load(open(jj('data', 'SVM_model.model'), 'rb'))
    prediction = SVM_model.predict(one_dim_feature_vector)

    print("\n")
    print("Człowiek" if prediction[0] == 1 else "Rzecz")
