from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import scipy.stats as stats
 
# Type of Cross Validation
class Validation:
    FOLD10 = 1
    LOSOCV = 2


# Folder path
DATASET_FLD = "3hx58k232n-4/"
DATA_FLD = "data/"
FOLDS_FLD = "temp/"

EDF_FILE_PATH = DATASET_FLD + 'Normal_Subject_'
TXT_FILE_PATH = DATASET_FLD + 'PSG_Outputs/PSG_Output_Normal_Subject_'

N_FTS = 41  # nr of features
N_FILES = 8  # nr of files
N_DIV = 9  # nr of 30s divisions in window. #3 5 7 9
sr = 256  # sampling rate (Hz)
NR_STAGES = 3  # sleep stages # 4 3
method_CV = Validation.FOLD10  # cross validation
PLOT = False  # plot ecg; confusion matrix
DEBUG = True  # information logs
FT_SELECTION = False  # feature selection
TRAIN_TEST_SIZE = 0.25

# Features
fts_name = ["nni_counter", "nni_mean", "nni_min", "nni_max", "nni_diff_mean", "nni_diff_min",
            "nni_diff_max", "hr_mean", "hr_min", "hr_max", "hr_std", "sdnn", "rmssd", "sdsd", "nn50",
            "pnn50", "nn20", "pnn20", "lomb_peak_vlf", "lomb_peak_lf", "lomb_peak_hf", "lomb_abs_vlf",
            "lomb_abs_lf", "lomb_abs_hf", "lomb_rel_vlf", "lomb_rel_lf", "lomb_rel_hf", "lomb_log_vlf",
            "lomb_log_lf", "lomb_log_hf", "lomb_norm_lf", "lomb_norm_hf", "lomb_ratio", "lomb_total",
            "sd1", "sd2", "sd_ratio", "ellipse_area", "sampen", "dfa_alpha1", "dfa_alpha2"]

# Classifiers
models = [
    'LDA',
    'RF',
    'KNN',
    'SVC',
    'LGBM',
    'XGB',
    'LSTM'
]

clfs = [
    LinearDiscriminantAnalysis(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(),
    LGBMClassifier(),
    XGBClassifier(),
    None
]


LDAParameters1 = {
    'solver' : ['svd', 'lsqr', 'eigen']
}

RandomForestParameters1 = {
    'n_estimators':[10,100,1000,1500],
    'max_features': ['sqrt' , 'log2'],
    'max_depth':[20, 50, 70, 100],
}

SVCParameters1 = {
  'kernel': ['linear', 'rbf'],
  'C': [1, 10, 100],
  'gamma': ['scale','auto'],
  'class_weight':[None,'balanced']
}

KNeighboursParameters1 = {
    'n_neighbors':[3, 5, 11, 15, 19],
    'weights':['uniform', 'distance'],
    'metric':['euclidean', 'manhattan']
}

LightGBMParameters1 = {
  'boosting_type': ['gbdt', 'dart', 'goss'],
  'max_depth': [5],
  'num_leaves': [14, 21, 28],
  'min_child_samples': [100, 200, 500],
  'objective': ['binary']
}

LightGBMParameters2 = {
  'boosting_type': ['gbdt', 'dart', 'goss'],
  'max_depth': [7],
  'num_leaves': [30, 70, 110],
  'min_child_samples': [100, 200, 500],
  'objective': ['binary']
}

LightGBMParameters3 = {
  'boosting_type': ['gbdt', 'dart', 'goss'],
  'max_depth': [9],
  'num_leaves': [50, 250, 450],
  'min_child_samples': [100, 200, 500],
  'objective': ['binary']
}
LightGBMParameters4 = {
  'boosting_type': ['rf'],
  'max_depth': [9, 7, 5],
  'num_leaves': [50, 30, 70, 110],
  'min_child_samples': [100, 200, 300],
  'objective': ['binary'],
  'bagging_fraction': np.linspace(0.1, 1, 5, endpoint=False).tolist(),
  'bagging_freq': np.linspace(1, 100, 5, endpoint=False, dtype=int).tolist()
}

XGBoostParameters1 = {
   'booster': ['gbtree'],
   'learning_rate': [0.3],
   'gamma': [0],
   'max_depth': [10,15],
   'min_child_weight': [0.5, 1],
   'max_delta_step': [0, 1],
   'subsample': [0.7, 1],
   'sampling_method': ["uniform"],
   'tree_method': ['auto']
}

XGBoostParameters2 = {
  'booster': ['dart'],
  'sample_type': ['uniform', 'weighted'],
  'normalize_type': ['tree', 'forest'],
  'rate_drop': [0.3]
}

XGBoostParameters3 = {
    'booster': ['gblinear'],
    'lambda': np.linspace(4, 10, 3, endpoint=True, dtype=int).tolist(),
    'alpha': np.linspace(4, 10, 3, endpoint=True, dtype=int).tolist(),
    'updater': ['coord_descent'],
    'feature_selector': ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty'],
}

XGBoostParameters4 = {
    'booster': ['gblinear'],
    'lambda': np.linspace(0, 10, 5, endpoint=True, dtype=int).tolist(),
    'alpha': np.linspace(0, 10, 5, endpoint=True, dtype=int).tolist(),
    'updater': ['shotgun'],
    'feature_selector': ['cyclic', 'shuffle'],
}

XGBoostParameters5 = {
  'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

decisionTreeParameters = {
    'criterion': ["gini", "entropy"],
    'splitter': ["best", "random"],
    'max_depth': np.linspace(1, 10, 10, endpoint=True).tolist(),
    'min_samples_split' : np.linspace(0.1, 1.0, 10, endpoint=True).tolist(),
    'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True).tolist(),
    'max_features': ["auto", "sqrt", "log2", None],
}


params = [
    # LDAParameters1,
    RandomForestParameters1,
    # KNeighboursParameters1,
    # SVCParameters1,
    # LightGBMParameters4
    # XGBoostParameters1, #XGBoostParameters2, XGBoostParameters3, XGBoostParameters4
]

best_params_list = [
  {
    'solver' : 'svd'
  },
  {
   'max_depth': 100,
   'max_features': 'sqrt',
   'n_estimators': 1000
  },
  {
    'metric': 'manhattan',
    'n_neighbors': 3,
    'weights': 'distance'
  },
  {
    'C': 100,
    'gamma': 1,
    'kernel': 'rbf',
    'class_weight': None
  },
  {
    'boosting_type': 'gbdt', 
    'max_depth': 9, 
    'min_child_samples': 100, 
    'num_leaves': 250, 
    'objective': 'binary'
  },
  {
    'booster': 'gbtree', 
    'gamma': 0, 
    'learning_rate': 0.3, 
    'max_delta_step': 0, 
    'max_depth': 10, 
    'min_child_weight': 0.5, 
    'sampling_method': 'uniform', 
    'subsample': 1, 
    'tree_method': 'auto'
  }, 
  {}
]