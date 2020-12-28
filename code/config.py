from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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

# Features
fts_name = ["nni_counter", "nni_mean", "nni_min", "nni_max", "nni_diff_mean", "nni_diff_min",
            "nni_diff_max", "hr_mean", "hr_min", "hr_max", "hr_std", "sdnn", "rmssd", "sdsd", "nn50",
            "pnn50", "nn20", "pnn20", "lomb_peak_vlf", "lomb_peak_lf", "lomb_peak_hf", "lomb_abs_vlf",
            "lomb_abs_lf", "lomb_abs_hf", "lomb_rel_vlf", "lomb_rel_lf", "lomb_rel_hf", "lomb_log_vlf",
            "lomb_log_lf", "lomb_log_hf", "lomb_norm_lf", "lomb_norm_hf", "lomb_ratio", "lomb_total",
            "sd1", "sd2", "sd_ratio", "ellipse_area", "sampen", "dfa_alpha1", "dfa_alpha2"]

# Classifiers
models = [
    # 'LDA',
    # 'RF',
    # 'SVC',
    # 'KNN',
     'LGBM',
    # 'XGB',
]

clfs = [
    # LinearDiscriminantAnalysis(),
    # RandomForestClassifier(max_depth=20),
    # SVC(class_weight='balanced'),
    # KNeighborsClassifier(n_neighbors=15),
     LGBMClassifier(),
    # XGBClassifier()
]
