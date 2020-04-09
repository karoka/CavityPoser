# Global variable
import pandas as pd
import numpy as np
import sys
from sklearn.externals import joblib

def get_data(dataset, threshold = 6.0, info_col = list(range(1,6)), label_col = 2):
    dataset['label'] = dataset[label_col] <= threshold
    return dataset['label'].mean(), dataset[info_col], dataset['label'], dataset.drop(info_col+['label'], axis=1)

## Read in raw pocket FPs
def main():
    modelFile = sys.argv[1]
    featureFile = sys.argv[2]
    outFile = sys.argv[3]
    scalar = int(sys.argv[4])
    model_names = "MLP XGB RF".split(" ")

    vec_dum_cols = list(range(346, 350)) +  list(range(690, 694)) + [1034]

    feature = pd.read_csv(featureFile, delim_whitespace=True, header=None, index_col=0)
    if not scalar:
        feature = feature.drop(vec_dum_cols, axis=1)

    print("Total number of cavities:", len(feature))

    _, info, _, X = get_data(feature)

    # load model
    models = joblib.load(modelFile)

    # scalar
    scaler = models['scaler']
    X_scaled = scaler.transform(X)

    for model_name in model_names:
        classifier = models[model_name]

        y_prob = classifier.predict_proba(X_scaled)[:,1]
        info['pred_%s'%model_name] = np.round(y_prob, 3)
    print(outFile, info)
    info.to_csv(outFile)
    return 0

if __name__ == '__main__':
    main()
