import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it
from IPython.display import display
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from hmmlearn import hmm
from tslearn.metrics import dtw
import sklearn_crfsuite
from time import time
import json
import warnings
warnings.filterwarnings('ignore')


Keypoints_1_7_normalized = pd.read_json('Keypoints_1_8_normalized_New.json', orient='records')
pivot = Keypoints_1_7_normalized.pivot_table(index=['Gesture', 'Sub folder No.', 'Frame No.'], columns=['Joint'], values=['X', 'Y', 'Depth'])

def joint_pair_distance(row, pairs):
    x = row['X']
    y = row['Y']
    z = row['Depth']
    result = np.empty(0)
    for pair in pairs:
        distance = np.sqrt(
            (x[pair[0]] - x[pair[1]]) ** 2 + (y[pair[0]] - y[pair[1]]) ** 2 + (z[pair[0]] - z[pair[1]]) ** 2)
        result = np.append(result, distance)
    return result

pairs = list(it.combinations(np.arange(1, 8), 2))
joint_pair_distance_feature = pivot.apply(lambda x: joint_pair_distance(x, pairs), axis=1).rename('Feature')
joint_pair_distance_feature_normalized = joint_pair_distance_feature.apply(lambda x: x / np.max(x)).rename('Feature_normalized')

gesture_number = len(Keypoints_1_7_normalized['Gesture'].unique())
dataset = joint_pair_distance_feature_normalized.reset_index().pivot_table(index=['Gesture', 'Sub folder No.'], values=['Feature_normalized'], aggfunc=lambda x: np.vstack(x).tolist()).iloc[:, 0]


# HMM 80% - 20% dataset
def hmm_data_8020():
    msk = np.concatenate([np.random.rand(len(dataset.loc[i])) < 0.8 for i in np.arange(1, gesture_number + 1)])
    train_data_8020 = dataset.iloc[msk]
    test_data_8020 = dataset.iloc[~msk]
    print('80%-20% Train dataset\'s recording number', len(train_data_8020))
    print('80%-20% Test dataset\'s recording number', len(test_data_8020))

    test_label_8020 = test_data_8020.reset_index(0).iloc[:, 0].values
    print('80%-20% Test label:', test_label_8020)

    hmm_train_data_8020 = [np.vstack(train_data_8020.loc[i].values) for i in range(1, gesture_number + 1)]
    [print('80%%-20%% Gesture %d Train data shape: %s' % (i + 1, hmm_train_data_8020[i].shape)) for i in
     range(gesture_number)]
    print()
    hmm_train_lengths_8020 = [train_data_8020.loc[i].apply(len).values for i in range(1, gesture_number + 1)]
    [print('80%%-20%% Gesture %d Train sequence lengths: %s' % (i + 1, hmm_train_lengths_8020[i])) for i in
     range(gesture_number)]
    print()

    return hmm_train_data_8020, hmm_train_lengths_8020, test_data_8020, test_label_8020


# HMM Jack-Knife dataset
def hmm_data_jackknife(person):
    train_data_jackknife = pd.concat(
        [dataset[[dataset.index.values[i][0] == j for i in range(len(dataset))]].iloc[pd.np.r_[0:person*3, person*3+3:18]] for j in
         range(1, gesture_number + 1)])
    test_data_jackknife = pd.concat(
        [dataset[[dataset.index.values[i][0] == j for i in range(len(dataset))]].iloc[person*3:person*3+3] for j in
         range(1, gesture_number + 1)])
    print('Jack-Knife Train dataset\'s recording number', len(train_data_jackknife))
    print('Jack-Knife Test dataset\'s recording number', len(test_data_jackknife))

    test_label_jackknife = test_data_jackknife.reset_index(0).iloc[:, 0].values
    print('Jack-Knife Test label:', test_label_jackknife)

    hmm_train_data_jackknife = [np.vstack(train_data_jackknife.loc[i].values) for i in range(1, gesture_number + 1)]
    [print('Jack-Knife Gesture %d Train data shape: %s' % (i + 1, hmm_train_data_jackknife[i].shape)) for i in
     range(gesture_number)]
    print()
    hmm_train_lengths_jackknife = [train_data_jackknife.loc[i].apply(len).values for i in range(1, gesture_number + 1)]
    [print('Jack-Knife Gesture %d Train sequence lengths: %s' % (i + 1, hmm_train_lengths_jackknife[i])) for i in
     range(gesture_number)]
    print()

    return hmm_train_data_jackknife, hmm_train_lengths_jackknife, test_data_jackknife, test_label_jackknife


# HMM Training and Testing
def hmm_train_test(epoch_n, validation_type, hmm_covariance_type, hmm_state_range):
    print('------------------------------- Start HMM Training and Testing of ', validation_type,
          ' validation, ', hmm_covariance_type, ' covariance -------------------------------')

    accuracy_matrix = np.empty((0,len(hmm_state_range)))

    for epoch in range(epoch_n):
        print('Start of epoch ', epoch, '\n')

        if validation_type == '8020': hmm_train_data, hmm_train_lengths, test_data, test_label = hmm_data_8020()
        if validation_type == 'jackknife': hmm_train_data, hmm_train_lengths, test_data, test_label = hmm_data_jackknife(epoch)

        accuracy = np.empty((0))
        run_time = np.empty((0))

        for state in hmm_state_range:
            hmm_model = []
            # Train a HMM model using time sequences of training, train one HMM model for each gesture
            for i in range(gesture_number):
                hmm_model.append(hmm.GaussianHMM(n_components=state, covariance_type=hmm_covariance_type).fit(hmm_train_data[i], lengths=hmm_train_lengths[i]))

            # After training HMM models, use them to classify each time sequence of testing
            scores = np.empty((0, gesture_number))

            start_time = time()
            for sequence in test_data.values:
                score = np.empty(gesture_number)
                for k in range(gesture_number):
                    try:
                        score[k] = hmm_model[k].score(sequence)
                    except:
                        score[k] = -99999
                scores = np.vstack((scores, score))
            test_predict = np.argmax(scores, axis=1) + 1
            end_time = time()

            run_time = np.append(run_time, (end_time - start_time) / len(test_label))
            accuracy = np.append(accuracy, np.sum(test_predict == test_label) / len(test_label))
            print('HMM state = %d,\tAccuracy = %.3f,\tRun time = %.3f\nLabel:\t\t%s\nPrediction:\t%s\n' % (state, accuracy[-1], run_time[-1], test_label, test_predict))

        accuracy_matrix = np.vstack((accuracy_matrix, accuracy.reshape(1, -1)))

        best_accuracy = np.max(accuracy)
        best_hmm_state = hmm_state_range[np.nonzero(accuracy == best_accuracy)]
        print('End of epoch %d\t\tBest accuracy = %.3f,\t\tHMM state of best accuracy = %s\n' % (epoch, best_accuracy, best_hmm_state))


    accuracy_mean = np.mean(accuracy_matrix, axis=0)
    print('------------------------------- End HMM Training and Testing of ', validation_type,
              ' validation, ', hmm_covariance_type, ' covariance -------------------------------')
    print('HMM states =    \t%s\nMean accuracy = \t%s\n' % (hmm_state_range, accuracy_mean))

    return accuracy_mean



state_range = np.arange(1, 27, 2)

accuracy_mean_8020_diag = hmm_train_test(epoch_n=6, validation_type='8020', hmm_covariance_type='diag', hmm_state_range=state_range)
accuracy_mean_8020_full = hmm_train_test(epoch_n=6, validation_type='8020', hmm_covariance_type='full', hmm_state_range=state_range)
accuracy_mean_jackknife_diag = hmm_train_test(epoch_n=6, validation_type='jackknife', hmm_covariance_type='diag', hmm_state_range=state_range)
accuracy_mean_jackknife_full = hmm_train_test(epoch_n=6, validation_type='jackknife', hmm_covariance_type='full', hmm_state_range=state_range)