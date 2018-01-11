import argparse
import pickle

import sklearn
from canny import Canny
from settings import WINDOW_SIZE
from utils import jj, open_image, resize_image_to_nxn_square, reshape_to_on_dim_vector, convert_to_array
import cv2
import matplotlib.pyplot as plt

# TODO: Add some graphical interface
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', help='Input file name. Default = data.txt', default=jj('data', 'test.jpg'))
    args = p.parse_args()
    SVM_model = pickle.load(open(jj('data', 'SVM_model.model'), 'rb'))

    # Open image and resize it
    img = open_image(args.input)

    results = []
    for i in range(1, 6):
        img = resize_image_to_nxn_square(img, i * WINDOW_SIZE)

        # Canny features extraction
        features_extractor = Canny(img, 50, 110)
        feature_matrix = features_extractor.get_features()
        # plt.imshow(feature_matrix)
        # plt.show()

        # im = cv2.Canny(img, 120, 220)
        # feature_matrix = convert_to_array(img)
        # plt.imshow(im)
        # plt.show()

        for row in range(2*i-1):
            for col in range(2*i-1):
                window = feature_matrix[int(row/2 * WINDOW_SIZE):int((row/2 + 1) * WINDOW_SIZE),
                                        int(col/2 * WINDOW_SIZE):int((col/2 + 1) * WINDOW_SIZE)]
                # plt.imshow(window)
                # plt.show()

                # Reshape to one dim
                one_dim_feature_vector = reshape_to_on_dim_vector(window, window.shape[0])

                # TODO: Implement SVM learning, get rid of pickle
                # Perform classification using SVM or Logistic Regression
                prediction = SVM_model.predict(one_dim_feature_vector)
                if prediction[0] == 1:
                    results.append(window)

    for image in results:
        plt.imshow(image)
        plt.show()

    # print("\n")
    # print("Człowiek" if prediction[0] == 1 else "Rzecz")
    # detect_multi_scale opencv
