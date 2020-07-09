import numpy as np
import pandas as pd
import collections
import random

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from time import time
from sklearn.metrics import classification_report
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


def load_data(home, scalar=1):
    vec_dum_cols = list(range(346, 350)) +  list(range(690, 694)) + [1034]
    if scalar:
        rawdata = "%s/features/rna-ligand-pocket.csv" %home
    else:
        rawdata = "%s/vec_features/rna-ligand-pocket.csv" %home
    dataset = pd.read_csv(rawdata, delim_whitespace=True, header=None, index_col=0)
    if not scalar:
        dataset = dataset.drop(vec_dum_cols, axis=1)
    rnas = set(dataset.index)
    print("Total number of RNAS:", len(rnas), "Total number of samples:", len(dataset))
    print("Number of positive samples:", sum(dataset[2] < 6))
    print("Number of negative samples:", sum(dataset[2] > 6))
    return rnas, dataset

def select_subset(dataset, subset = ("1AJU"), id_index = 0, include = True):
    '''
    Select subset of RNAs
    '''
    if include:
        return dataset[dataset.index.isin(subset)]
    else:
        return dataset[~dataset.index.isin(subset)]

def partition(rnas, dataset):
    test1_id = set(['1F1T', '2B57', '2G5K', '2O3W', '2XNW', '2YDH', '3FU2', '3LA5',
                    '3MUM', '3NPN', '3Q50', '3SD3', '3SLM', '4AOB', '4ERJ', '4FE5',
                    '4JF2', '4KQY', '4L81', '4LX5', '4NYA', '4XWF', '4YB0', '5C7W', '5KPY'])
    test2_id = set(['1AJU', '1AKX', '1AM0', '1BYJ', '1EHT', '1EI2', '1EVV', '1FMN',
                    '1KOC', '1KOD', '1LVJ', '1NEM', '1O9M', '1PBR', '1Q8N', '1QD3',
                    '1TOB', '1UTS', '1UUD', '1UUI', '2TOB'])
    exclude_ires = set(['3TZR', '1P5M', '1P5O', '1P5P', '2AGN', '2KTZ', '2KU0', '2NOK',
                        '2PN3', '2PN4', '3P59', '3TZR', '4UJC', '4UJD', '5A2Q', '5FLX',
                        '6IP6', '6IP8'])
    train_id = rnas - (test1_id | test2_id | exclude_ires)

    train = select_subset(dataset, train_id)
    test1 = select_subset(dataset, test1_id)
    test2 = select_subset(dataset, test2_id)
    test3 = select_subset(dataset, exclude_ires)

    print("Number of training samples: %d\nNumber of test 1 samples: %d\nNumber of test 2 samples: %d\nNumber of test 3 samples: %d"
          %(len(train), len(test1), len(test2), len(test3)))
    return train, test1, test2, test3


def get_data(dataset, threshold = 6.0, info_col = list(range(1,6)), label_col = 2):
    '''
    Input: pandas dataframe
    Params:
        threshold: max RMSD between ligand and native cavity
        info_col: columns not used as features
        label_col:
    Return: fraction of postive samples, cavity info, label (native/decoy), fingerprints
    '''
    dataset['label'] = dataset[label_col] <= threshold
    return dataset['label'].mean(), dataset[info_col], dataset['label'], dataset.drop(info_col+['label'], axis=1)


def subset_features(trainX, featureNames, etas=('2','4', '8', '16')):
    '''
    Select subset of features by eta values
    Return: index of select columns
    '''
    nameMaps = dict(zip(featureNames, trainX))
    return [nameMaps[feat] for feat in featureNames if feat.startswith(etas)]


def preprocess(home, trainX, testXs, trainy, add_dummy = 0,
               subset = False, subset_etas=('2','4', '8', '16'),
               scale=True):
    '''
    1. Add dummy samples (0-vector) to trainX. add_dummy: number of samples to add
    2. Subset features by eta
    3. Scale features using StandardScaler
    Input:
        trainX: training set features
        testX: a list of testsets features, e.g. [test1X, test2X, ...]
    Return: trainX, testXs
    '''

    # add dummy sample - empty box
    if add_dummy:
        for i in range(add_dummy):
            trainX.loc[len(trainX)] = 0
            trainy.loc[len(trainy)] = False

    # subset features
    if subset:
        featureNames = pd.read_csv("%s/features/feature_header.txt" %home, delim_whitespace=True).columns[3:]
        sel_cols = subset_features(trainX, featureNames, etas=subset_etas)
        trainX = trainX[sel_cols]
        for i, testX in enumerate(testXs):
            testXs[i] = testX[sel_cols]

    # setup scaler
    scaler = None
    if scale:
        scaler = StandardScaler()
        scaler.fit(trainX)

        # transform input
        trainX = scaler.transform(trainX)
        for i, testX in enumerate(testXs):
            testXs[i] = scaler.transform(testX)

    _, nfeatures = trainX.shape
    print("Number of features:", nfeatures)
    return trainX, testXs, scaler


def search_pipeline(X_train_data, y_train_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       search_mode = 'GridSearchCV', n_iterations = 0):
    X_train_data, y_train_data = shuffle(X_train_data, y_train_data, random_state=0)
    fitted_model = None

    if(search_mode == 'GridSearchCV'):
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring=scoring_fit,
            verbose=2
        )
        fitted_model = gs.fit(X_train_data, y_train_data)

    elif (search_mode == 'RandomizedSearchCV'):
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            cv=cv,
            n_iter=n_iterations,
            n_jobs=-1,
            scoring=scoring_fit,
            verbose=0
        )
        fitted_model = rs.fit(X_train_data, y_train_data)

    if (fitted_model != None):
        val_score = fitted_model.best_score_
        train_score = fitted_model.score(X_train_data, y_train_data)
        return fitted_model, train_score, val_score

# set model and parameter grid for models
def train_LR():
    model = LogisticRegression(max_iter=1e4, tol=1e-4)
    param_grid = {
        'class_weight': ['balanced', None],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    return model, param_grid

def train_RF(nfeatures):
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': range(20, 500, 5),
        'max_features': ['auto', 'sqrt'],
        'max_depth': range(2, int(np.log(nfeatures)), 2),
        'min_samples_split': range(2, 10, 3), # at least 2
        'min_samples_leaf': range(1, 10, 3),
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    return model, param_grid

def train_XGB(nfeatures):
    model = xgb.XGBClassifier()
    param_grid = {
        'n_estimators': range(20, 500, 5),
        'colsample_bytree': [0.7, 0.8],
        'max_depth': range(2, int(np.log(nfeatures)), 2),
        'reg_alpha': [1.1, 1.2, 1.3],
        'reg_lambda': [1.1, 1.2, 1.3],
        'subsample': [0.7, 0.8, 0.9]
    }
    return model, param_grid

def train_MLP(min_size = 5, max_size = 100):
    model = MLPClassifier(max_iter=100)
    param_grid = {
        'alpha': [1],
        'max_iter': [1000],
        'solver': ['adam'],
        'activation': ['relu']
    }
    param_grid = {
        'hidden_layer_sizes': [(sp_randint.rvs(min_size, max_size),sp_randint.rvs(min_size, max_size),sp_randint.rvs(min_size, max_size),sp_randint.rvs(min_size, max_size)),
                               (sp_randint.rvs(min_size, max_size),sp_randint.rvs(min_size, max_size),sp_randint.rvs(min_size, max_size)),
                               (sp_randint.rvs(min_size, max_size),sp_randint.rvs(min_size, max_size)), (sp_randint.rvs(min_size, max_size),)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': sp_expon(scale=.01),
        'learning_rate': ['constant','adaptive'],
        'learning_rate_init': sp_expon(scale=.001),
    }
    return model, param_grid

# train models
def train(trainX, trainy, model_names = ['XGB', 'RF', 'MLP']):

    models = {}
    random.seed(0)
    nfeatures = trainX.shape[1]

    # model_names = ['LR', 'XGB', 'RF', 'MLP']
    for model_name in model_names:
        if model_name == 'XGB':
            model, param_grid = train_XGB(nfeatures)
        if model_name == 'RF':
            model, param_grid = train_RF(nfeatures)
        if model_name == 'MLP':
            model, param_grid = train_MLP()
        if model_name == "LR":
            model, param_grid = train_LR()

        # train model
        model, train_score, val_score = search_pipeline(trainX, np.int_(trainy), model,
                                      param_grid, cv=5, search_mode = "RandomizedSearchCV", n_iterations = 100)

        # attach to model dictionaries
        models[model_name] = model

        # Assess performance
        print("%s: train score %.3f validation score %.3f" %(model_name, train_score, val_score))
    return models


def test(reshome, models, scaler, testys, testXs, info_tests, write_file = False):
    '''
    Assess models performance on test set
    '''

    scores = {}
    y_probs = {}

    for j, testX in enumerate(testXs):
        y_true = np.int_(testys[j])
        for i, model_name in enumerate(list(models.keys())):
            if model_name == "scaler":
                continue
            # predict testX
            y_prob = models[model_name].predict_proba(testX)[:,1]
            # udpate info_tests
            info_tests[j]['pred_%s'%model_name] = np.round(y_prob, 3)
            # compute auc score
            score = roc_auc_score(y_true, y_prob, average='macro', sample_weight=None, max_fpr=None)
            scores[model_name] = scores.get(model_name, []) + [score]
            y_probs[j] = y_probs.get(j, 0) + y_prob
        # combined prediction
        info_tests[j]['pred_COMP'] = np.round(y_probs[j], 3)
        if write_file:
            info_tests[j].to_csv("%s/test_set%d_results.csv" %(reshome, j+1))
        # combined scores
        score = roc_auc_score(y_true, y_probs[j], average='macro', sample_weight=None, max_fpr=None)
        scores["combined"] = scores.get("combined", []) + [score]

    for model in scores:
        print(model, scores[model])


def save_model(modelhome, models, scaler, label=""):
    # save model
    models['scaler'] = scaler
    model_fn = "%s/%smodels.pkl" %(modelhome, label)
    joblib.dump(models, model_fn, compress=3)

def load_model(modelhome, label=""):
    # save model
    model_fn = "%s/%smodels.pkl" %(modelhome, label)
    return joblib.load(model_fn)

def analyze_data(dataset):
    import matplotlib
    nperRNA = collections.Counter(dataset.index)
    freqs = {}
    for rna, n in nperRNA.items():
        freqs[n] = freqs.get(n, []) + [rna]
    for n in sorted(freqs):
        print("Number of cavities ", n, freqs[n])
