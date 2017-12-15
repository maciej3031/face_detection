import argparse
import pickle

from canny import Canny
from utils import jj

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', help='Input file name. Default = data.txt', default=jj('data', 'example2.jpg'))
    args = p.parse_args()

    features_extractor = Canny()
    feature_vector = features_extractor.get_features(args.input)

    SVM_model = pickle.load(open(jj('data', 'lr_model.model'), 'rb'))
    prediction = SVM_model.predict(feature_vector)

    print("Człowiek" if prediction[0] == 1 else "Rzecz")
