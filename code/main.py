from config import *

# utils
from collections import Counter
import numpy as np
import pandas as pd
import argparse
import itertools
import statistics as stat

# edf reader
from pyedflib import EdfReader

# hrv analysis
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl

# ecg peaks
from biosppy.signals.ecg import ecg as ecg_signal

# plot ecg
from bokeh.plotting import figure, show, output_file

# plot confusion matrix
import matplotlib.pyplot as plt

# oversampling
from imblearn.over_sampling import SVMSMOTE

# scaling
from sklearn import preprocessing

# cross validation
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

# evaluation metrics
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, confusion_matrix, \
    precision_recall_fscore_support, cohen_kappa_score

# ecg output
output_file("test.html")


# -------------------------------------------------------------

def get_features(r_peaks):
    """
    Compute features for a time window
    :param r_peaks: list of peaks
    :return: list of features and if valid
    """

    nni = np.diff(r_peaks, axis=0).ravel()  # nni intervals (s)

    fts = []  # features list

    # time domain
    fts.extend(td.nni_parameters(nni=nni))  # ms
    fts.extend(td.nni_differences_parameters(nni=nni))
    fts.extend(td.hr_parameters(nni=nni))
    fts.extend(td.sdnn(nni=nni))
    fts.extend(td.rmssd(nni=nni))
    fts.extend(td.sdsd(nni=nni))
    fts.extend(td.nn50(nni=nni))
    fts.extend(td.nn20(nni=nni))

    # frequency
    fdf = fd.lomb_psd(nni=nni, show=False)

    tuples = [fdf[x] for x in ["lomb_peak", "lomb_abs", "lomb_rel", "lomb_log", "lomb_norm"]]
    fts.extend([item for t in tuples for item in t])
    fts.extend(fdf[x] for x in ["lomb_ratio", "lomb_total"])

    # non linear
    nlf = nl.nonlinear(nni=nni)

    fts.extend(nlf[key] for key in nlf.keys() if key in fts_name)

    # check if valid
    valid = all([(x != "nan" and x != "-inf" and x != "inf") for x in fts])

    fts = np.asarray(fts, dtype="float32")
    fts = np.nan_to_num(fts)

    return valid, fts


def read_file(i):
    """
    Reads ECG file in EDF format.
    :param i: file number.
    :return: signal, stages, and reliability
    """

    print("File " + str(i) + "...")

    # read edf file
    f = EdfReader(EDF_FILE_PATH + "{:02d}".format(i) + '.edf')

    # get ecg signal
    ecg = f.readSignal(14)
    ecg = ecg * 0.001  # uV to mV

    f.close()

    # read sleep stages
    f = open(TXT_FILE_PATH + "{:02d}".format(i) + '/Sleep Profile.txt', "r")

    # get stages
    stages = []
    for line in itertools.islice(f, 8, None):
        stages.append(line.split("; ")[1].rstrip())

    f.close()

    # read reliability
    f = open(TXT_FILE_PATH + "{:02d}".format(i) + '/Sleep Profile Reliability.txt', "r")
    reliability = []
    for line in itertools.islice(f, 7, None):
        reliability.append(line.split("; ")[1].rstrip())

    f.close()

    # stage 4 to 3
    stages = np.char.replace(stages, 'Stage 4', 'Stage 3')

    # synchronize in time
    nr_windows = int((len(ecg) / sr) / 30)
    if nr_windows < len(stages):
        stages = stages[:-1]

    assert nr_windows == len(stages)

    return ecg, stages, reliability


def get_windows(ecg):
    """
    Compute windows of indices.
    :param ecg: ecg signal
    :return: windows
    """

    window_size = N_DIV * 30  # s
    n = sr * window_size  # samples
    nr_windows = int((len(ecg) / n) * N_DIV) - (N_DIV - 1)

    windows = []

    for idx, win in enumerate(range(0, len(ecg), int(n / N_DIV))):
        if idx < nr_windows:
            windows.append(list(range(win, win + n)))

    return windows


def plot_ecg(t, s, peaks, label):
    """
    Plot ecg window.
    :param t: time axis
    :param s: signal
    :param peaks: peaks
    :param label: sleep stage
    """

    p = figure(plot_width=700, plot_height=285)
    p.grid.visible = False
    p.circle(t[peaks], s[peaks], size=6, legend_label='peak', color='#5F5F5F')
    p.line(t, s, legend_label='ecg', color='#476D99')
    p.title.text = "ECG"
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Time (s)'
    p.yaxis.axis_label = 'ECG (mV)'
    p.title.text = label
    show(p)


def iterate_windows(ecg, windows, stages, reliability):
    """
    Iterate windows and compute features.
    :param ecg: ecg file
    :param windows: ecg windows
    :param stages: sleep stages
    :param reliability: sleep stages reliability
    :return: list of features, classes, and annotations
    """

    # movement epochs
    idx_mov = [i for i in range(len(stages)) if stages[i] == "Movement"]

    # stages to remove (sides)
    nr_side = int((N_DIV - 1) / 2)

    # store results
    features = np.array([]).reshape(0, N_FTS)
    annotations = []

    for idx, s in enumerate(ecg[windows]):

        print(" ", idx + 1, "/", len(windows), end='\r')

        # detect ecg peaks
        (t, s, r_peaks) = ecg_signal(signal=s, sampling_rate=sr, show=False)[0:3]

        # window has any movement epoch
        mvt_label = any(x in list(range(idx, idx + N_DIV)) for x in idx_mov)

        if PLOT:
            label = stages[idx + nr_side] + " : " + str(not mvt_label)
            plot_ecg(t, s, r_peaks, label)
            input("Press Enter to continue...")

        # compute features
        (valid, ft) = get_features(t[r_peaks])

        # get annotations
        annotations.append((not mvt_label, valid, not reliability[idx + nr_side] == "<35% Reliability"))

        # append features
        features = np.vstack((features, ft))

    # remove sides
    classes = stages[nr_side: len(stages) - nr_side]

    assert len(features) == len(classes)

    return features, classes, annotations


def compute_features(rng):
    """
    Compute and save features.
    :param rng: range of files
    """

    print("Computing features...")

    for i in rng:
        # read files
        (ecg, stages, reliability) = read_file(i)

        # get windows
        windows = get_windows(ecg)

        # compute features
        (features, classes, params) = iterate_windows(ecg, windows, stages, reliability)

        # save (iteratively)
        np.save(DATA_FLD + 'data_x_' + str(i) + "_" + str(N_DIV) + '.npy', np.asarray(features))
        np.save(DATA_FLD + 'data_y_' + str(i) + "_" + str(N_DIV) + '.npy', np.asarray(classes))
        np.save(DATA_FLD + 'params_' + str(i) + "_" + str(N_DIV) + '.npy', np.asarray(params))


def load_array(rng):
    """
    Load array with all subjects
    :param rng: range of files
    :return: X and Y
    """

    print("Loading data 10CV...")

    X = np.array([], dtype="float32").reshape(0, N_FTS)
    y = np.array([])
    params = np.array([]).reshape(0, 3)

    for i in rng:
        _X = np.load(DATA_FLD + 'data_x_' + str(i) + "_" + str(N_DIV) + '.npy')
        _y = np.load(DATA_FLD + 'data_y_' + str(i) + "_" + str(N_DIV) + '.npy')
        _params = np.load(DATA_FLD + 'params_' + str(i) + "_" + str(N_DIV) + '.npy')

        # append
        X = np.concatenate((X, _X))
        y = np.concatenate((y, _y))
        params = np.concatenate((params, _params))

    X, y = filter_params(X, y, params)

    return X, y


def load_subjects(rng):
    """
    Load subjects
    :param rng: range of files
    :return: X and Y
    """

    print("Loading data LOSOCV...")

    X, y = [], []

    for i in rng:
        _X = np.load(DATA_FLD + 'data_x_' + str(i) + "_" + str(N_DIV) + '.npy')
        _y = np.load(DATA_FLD + 'data_y_' + str(i) + "_" + str(N_DIV) + '.npy')
        _params = np.load(DATA_FLD + 'params_' + str(i) + "_" + str(N_DIV) + '.npy')

        _X, _y = filter_params(_X, _y, _params)

        X.append(_X)
        y.append(_y)

    X = np.asarray(X, dtype="object")
    y = np.asarray(y, dtype="object")

    return X, y


def filter_params(X, y, params):
    """
    Filter X and Y according to the parameters
    :param X: features
    :param y: label
    :param params: parameters
    :return: filtered X and Y
    """

    if NR_STAGES == 4:

        y = np.char.replace(y, 'Stage 1', 'Light')
        y = np.char.replace(y, 'Stage 2', 'Light')
        y = np.char.replace(y, 'Stage 3', 'Deep')

    elif NR_STAGES == 3:
        y = ['NREM' if (stage == 'Stage 1' or stage == 'Stage 2' or stage == 'Stage 3') else stage for stage in y]

    # Filter by parameters
    MVT = 0  # movement
    VALID = 0  # valid
    RLB = 0  # reliability

    bf_len = len(X)

    if params is not None:
        idxs = set()
        idxs.update([i for i in range(len(params)) if
                     (params[i][0] == 1 if MVT else True)
                     and
                     (params[i][1] == 1 if VALID else True)
                     and
                     (params[i][2] == 1 if RLB else True)
                     ])

        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

    # Remove movement labels
    if (not MVT) or (params is None):
        idxs = [i for i in range(len(y)) if y[i] != "Movement"]
        X = [X[i] for i in idxs]
        y = [y[i] for i in idxs]

    if DEBUG:
        removed = bf_len - len(X)
        print("[ Filter: removed:", removed, "({:.2f}%)".format((removed / bf_len) * 100) + " ]")

    X = np.asarray(X)
    y = np.asarray(y)

    assert len(X) == len(y)

    return X, y


def ft_selection(train_df, test_df):
    """
    Feature selection
    :param train_df: train dataframe
    :param test_df: test dataframe
    :return: test dataframe
    """

    if DEBUG:
        print("\n[ Feature selection ]")

    threshold = 0.9  # threshold of correlation
    col_corr = set()  # names of deleted columns
    corr_matrix = train_df.corr()  # correlation matrix

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                col_name = corr_matrix.columns[i]  # column name
                col_name2 = corr_matrix.columns[j]

                if DEBUG:
                    print("... Features correlated - " + col_name + ";" + col_name2)
                col_corr.add(col_name)

    if DEBUG:
        print("... Features not selected - " + str(col_corr))

    # delete
    train_df.drop(col_corr, axis=1, inplace=True)
    test_df.drop(col_corr, axis=1, inplace=True)


def compute_folds(X, y):
    """
    Compute and save cross validation folds
    :param X: features
    :param y: labels
    """

    print("Computing folds...")

    if method_CV == Validation.LOSOCV:
        skf = LeaveOneOut()
    else:
        skf = StratifiedKFold(n_splits=10, shuffle=True)

    # normalize features
    scaler = preprocessing.StandardScaler()

    idx = 0

    # split
    for train_idx, test_idx in skf.split(X, y):

        print("\rFold {} ...".format(idx + 1), end="")

        if method_CV == Validation.FOLD10:

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        else:

            X_, X_test = X[train_idx], X[test_idx][0]
            Y_, y_test = y[train_idx], y[test_idx][0]

            # concatenate subjects in train
            X_train = np.array([]).reshape(0, N_FTS)
            y_train = np.array([])

            for x_, y_ in zip(X_, Y_):
                X_train = np.concatenate((X_train, x_))
                y_train = np.concatenate((y_train, y_))

        # normalize
        X_train = scaler.fit_transform(X_train)  # fit only with X_train
        X_test = scaler.transform(X_test)

        # convert to dataFrame
        X_test = pd.DataFrame(data=X_test, columns=fts_name)
        X_train = pd.DataFrame(data=X_train, columns=fts_name)

        # feature selection
        if FT_SELECTION:
            ft_selection(X_train, X_test)
        else:
            # select only 7 features
            col_selected = {"nni_counter", "nni_mean", "nni_min", "nni_max", "nni_diff_mean", "nni_diff_max", "sdnn"}

            X_test = X_test[col_selected]
            X_train = X_train[col_selected]

        # oversampling
        if DEBUG:
            print("\n[ Oversampling ]")
            print('... Original dataset shape %s' % Counter(y_train))

        sm = SVMSMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)

        if DEBUG:
            print('... Resampled dataset shape %s' % Counter(y_train))

        # save
        np.save(FOLDS_FLD + 'y_train_' + str(idx) + '.npy', y_train)
        np.save(FOLDS_FLD + 'y_test_' + str(idx) + '.npy', y_test)
        X_train.to_pickle(FOLDS_FLD + 'X_train_' + str(idx) + '.pkl')
        X_test.to_pickle(FOLDS_FLD + 'X_test_' + str(idx) + '.pkl')

        idx += 1


def train_method(name, clf):
    """
    Trains method with folds
    :param name: classifier name
    :param clf: classifier
    """

    print("\nTraining algorithm "+name+"...")

    conf_matrix_array = []
    score_array = []
    sc = []
    kappa = []

    rng = (range(0, 10) if method_CV == Validation.FOLD10 else range(0, 11))

    for idx in rng:

        print("\rFold {} ...".format(idx + 1), end="")

        # load folds
        y_train = np.load(FOLDS_FLD + 'y_train_' + str(idx) + '.npy')
        y_test = np.load(FOLDS_FLD + 'y_test_' + str(idx) + '.npy')
        X_train = pd.read_pickle(FOLDS_FLD + 'X_train_' + str(idx) + '.pkl')
        X_test = pd.read_pickle(FOLDS_FLD + 'X_test_' + str(idx) + '.pkl')

        # fit train
        clf = clf.fit(X_train, y_train)

        # predict
        y_pred = clf.predict(X_test)

        # results
        rep = classification_report(y_test, y_pred)
        if DEBUG:
            print("\n", rep)

        # store
        sc.append(accuracy_score(y_test, y_pred))  # accuracy
        kappa.append(cohen_kappa_score(y_test, y_pred))  # kappa
        conf_matrix_array.append(confusion_matrix(y_test, y_pred, normalize='true'))  # matrix
        score_array.append(precision_recall_fscore_support(y_test, y_pred))  # metrics for each label

        if PLOT:  # confusion matrix
            disp_train = plot_confusion_matrix(clf, X_train, y_train, cmap=plt.cm.Blues, normalize='true')
            disp_train.ax_.set_title("Train")

            disp_test = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
            disp_test.ax_.set_title("Test")

            plt.show(block=False)
            input("\nPress Enter to continue...")
            plt.close()

    print("\nResults...")
    print("- Accuracy = {:.2f} ({:.2f})".format(stat.mean(sc), stat.stdev(sc)))
    print("- Kappa = {:.2f} ({:.2f})".format(stat.mean(kappa), stat.stdev(kappa)))

    print("- Confusion matrix (mean; std)")
    print(np.round(np.mean(conf_matrix_array, axis=0), 2))
    print(np.round(np.std(conf_matrix_array, axis=0), 2))

    # Sleep stages
    precision, recall, f_score, _ = np.mean(score_array, axis=0)
    precision_std, recall_std, f_score_std, _ = np.std(score_array, axis=0)

    print("- Stages recall")
    if NR_STAGES == 3:
        print("-- NREM = {:.2f} ({:.2f})".format(recall[0], recall_std[0]))
        print("-- REM = {:.2f} ({:.2f})".format(recall[1], recall_std[1]))
        print("-- Wake = {:.2f} ({:.2f})".format(recall[2], recall_std[2]))
    else:
        print("-- Deep = {:.2f} ({:.2f})".format(recall[0], recall_std[0]))
        print("-- Light = {:.2f} ({:.2f})".format(recall[1], recall_std[1]))
        print("-- Rem = {:.2f} ({:.2f})".format(recall[2], recall_std[2]))
        print("-- Wake = {:.2f} ({:.2f})".format(recall[3], recall_std[3]))


def run(train, folds):
    """
    Train methods
    :param train: compute features
    :param folds: compute folds
    """

    rng = range(1, N_FILES + 1)  # file range

    if train or folds:

        # compute ecg features
        if train:
            compute_features(rng)

        # load data and filter
        if method_CV == Validation.LOSOCV:
            (X, y) = load_subjects(rng)

        else:
            (X, y) = load_array(rng)

        # get folds
        compute_folds(X, y)

    # train algorithms
    for name, clf in zip(models, clfs):
        train_method(name, clf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sono ao Volante.")
    parser.add_argument('--train', help="Compute ECG features", action='store_true')
    parser.add_argument('--folds', help="Compute folds", action='store_true')
    args = parser.parse_args()

    run(args.train, args.folds)
