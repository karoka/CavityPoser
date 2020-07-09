import sys
from train_utils import *
import os

def main():
    scalar = True
    pretrain = int(sys.argv[1])
    home = os.getcwd()
    reshome = "%s/results" %home
    modelhome = "%s/models/" %home
    rnas, dataset = load_data(home, scalar = scalar)
    train_df, test1, test2, test3 = partition(rnas, dataset)

    frac_train, info_train, trainy, trainX = get_data(train_df)
    frac_test1, info_test1, test1y, test1X = get_data(test1)
    frac_test2, info_test2, test2y, test2X = get_data(test2)
    frac_test3, info_test3, test3y, test3X = get_data(test3)

    testXs = [test1X, test2X, test3X]
    info_tests = [info_test1, info_test2, info_test3]
    testys = [test1y, test2y, test3y]

    label = ""
    if not scalar:
        label = "vec_"

    if pretrain:
        models = load_model(modelhome, label=label)
        scaler = models['scaler']
        for i, testX in enumerate(testXs):
            testXs[i] = scaler.transform(testX)
    else:
        trainX, testXs, scaler = preprocess(home, trainX, testXs, trainy, add_dummy = 10, subset = False, scale = True)
        models = train(trainX, trainy)
        save_model(modelhome, models, scaler, label=label)

    test(reshome, models, scaler, testys, testXs, info_tests, write_file = True)

    return 0

if __name__ == '__main__':
    main()
