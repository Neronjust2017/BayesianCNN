from builtins import print
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scipy.io import arff
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import operator
import h5py
import glob
import scipy
from collections import Counter


def readucr(filename):
    data = np.loadtxt(filename)
    # 混洗 shuffle 默认打乱第一维度
    np.random.shuffle(data)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def readmts(filename):
    data = arff.loadarff(filename + '_TRAIN.arff')
    df = pd.DataFrame(data[0])
    values = df.values

    TrainSize = values.shape[0]
    SeriesLength = len(values[0][0][0])
    NumDimensions = values[0][0].size

    train_data = np.zeros((TrainSize, SeriesLength, NumDimensions))
    train_label = []
    for i in range(TrainSize):
        train_label.append(values[i][1].decode())
        for j in range(NumDimensions):
            a = list(values[i][0][j])
            train_data[i, :, j] = np.array(a, dtype=float)

    data = arff.loadarff(filename + '_TEST.arff')
    df = pd.DataFrame(data[0])
    values = df.values

    TestSize = values.shape[0]

    test_data = np.zeros((TestSize, SeriesLength, NumDimensions))
    test_label = []
    for i in range(TestSize):
        test_label.append(values[i][1].decode())
        for j in range(NumDimensions):
            a = list(values[i][0][j])
            test_data[i, :, j] = np.array(a, dtype=float)

    nb_classes = len(np.unique(np.concatenate((train_label, test_label), axis=0)))
    labels = train_label + test_label
    labels = list(set(labels))
    le = LabelEncoder().fit(labels)
    new_train_label = le.transform(train_label)
    new_test_label = le.transform(test_label)

    permutation = np.random.permutation(TrainSize)
    X_train = train_data[permutation, :, :]
    y_train = new_train_label[permutation]

    permutation = np.random.permutation(TestSize)
    X_test = test_data[permutation, :, :]
    y_test = new_test_label[permutation]

    return X_train, y_train, X_test, y_test, nb_classes


def readmts_uci_har(filename):  # file_name = 'datasets/uts_data/' + config.dataset.name

    path_train = filename + '/train/Inertial Signals/'
    file_train = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt', 'body_gyro_x_train.txt',
                  'body_gyro_y_train.txt', 'body_gyro_z_train.txt', 'total_acc_x_train.txt', 'total_acc_x_train.txt',
                  'total_acc_z_train.txt']
    list_train = []
    for i in range(9):
        data = np.loadtxt(path_train + file_train[i])
        list_train.append(data)

    length = list_train[0].shape[1]
    train_size = list_train[0].shape[0]

    train_data = np.zeros((train_size, length, 9))
    for i in range(train_size):
        for j in range(9):
            train_data[i, :, j] = list_train[j][i]

    train_label = np.loadtxt(filename + '/train/y_train.txt')

    path_test = filename + '/test/Inertial Signals/'
    file_test = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt', 'body_gyro_x_test.txt',
                 'body_gyro_y_test.txt', 'body_gyro_z_test.txt', 'total_acc_x_test.txt', 'total_acc_x_test.txt',
                 'total_acc_z_test.txt']
    list_test = []
    for i in range(9):
        data = np.loadtxt(path_test + file_test[i])
        list_test.append(data)

    length = list_test[0].shape[1]
    test_size = list_test[0].shape[0]

    test_data = np.zeros((test_size, length, 9))
    for i in range(test_size):
        for j in range(9):
            test_data[i, :, j] = list_test[j][i]

    test_label = np.loadtxt(filename + '/test/y_test.txt')

    permutation = np.random.permutation(train_size)
    X_train = train_data[permutation, :, :]
    y_train = train_label[permutation]

    permutation = np.random.permutation(test_size)
    X_test = test_data[permutation, :, :]
    y_test = test_label[permutation]

    return X_train, y_train, X_test, y_test


def readmts_ptb(filename):
    ## Loading time serie signals
    files = sorted(glob.glob(filename + "*.mat"))
    count = 0
    maxlen = -1
    minlen = 999

    # all_data = np.zeros((len(files), 200000))
    all_data = []
    for i in range(len(files)):
        # if i in index2del:
        #     continue
        f = files[i]
        record = f[:-4]
        record = record[-6:]
        # Loading
        mat_data = scipy.io.loadmat(f[:-4] + ".mat")
        print('Loading record {}'.format(record))
        data = mat_data['val'].squeeze()
        all_data.append(data)
        data = np.array(data)
        if len(data) > maxlen:
            maxlen = len(data)
        if len(data) < minlen:
            minlen = len(data)
    all_data = np.array(all_data)
    # normalize
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        data_tmp = (tmp_data - tmp_mean) / tmp_std
        data_tmp = data_tmp[:minlen]
        all_data[i] = data_tmp
    all_data = np.array(all_data)

    # WINDOW_SIZE = maxlen
    # ## 短数据处理
    #
    # for i in range(len(files)):
    #     offset = 0
    #     while offset == 0 or offset + WINDOW_SIZE < len(all_data[i]):
    #         slice = np.zeros(WINDOW_SIZE)
    #         if offset + WINDOW_SIZE < len(all_data[i]):
    #             slice[offset: offset + WINDOW_SIZE] = x_train[i][offset: offset + WINDOW_SIZE]
    #         else:
    #             slice[offset: offset + len(x_train[i])] = x_train[i][offset: len(x_train[i])]

    ## read label
    label_df = pd.read_csv(os.path.join(filename, 'REFERENCE-v3.csv'), header=None)
    label = label_df.iloc[:, 1].values
    all_label = []
    for i in label:

        # Myocardial infarction (368)
        if i == 'Myocardial infarction':
            all_label.append(0)

        # 'Healthy control': 80
        elif i == 'Healthy control':
            all_label.append(1)

        # 'Valvular heart disease': 6
        # elif i == 'Valvular heart disease':
        #     all_label.append(2)

        # 'Dysrhythmia': 16
        elif i == 'Dysrhythmia':
            all_label.append(2)

        # 'Heart failure (NYHA 2)': 1
        # elif i == 'Heart failure (NYHA 2)':
        #     all_label.append(4)

        # 'Heart failure (NYHA 3)': 1
        # elif i == 'Heart failure (NYHA 3)':
        #     all_label.append(5)

        # 'Heart failure (NYHA 4)': 1
        # elif i == 'Heart failure (NYHA 4)':
        #     all_label.append(6)

        # 'Palpitation': 1
        # elif i == 'Palpitation':
        #     all_label.append(7)

        # 'Cardiomyopathy': 17
        elif i == 'Cardiomyopathy':
            all_label.append(3)

        # 'Stable angina': 2
        # elif i == 'Stable angina':
        #     all_label.append(9)

        # 'Hypertrophy': 7
        # elif i == 'Hypertrophy':
        #     all_label.append(10)

        # 'Bundle branch block': 17
        elif i == 'Bundle branch block':
            all_label.append(4)

        # 'Unstable angina': 1
        # elif i == 'Unstable angina':
        #     all_label.append(12)

        # 'Myocarditis': 4
        # elif i == 'Myocarditis':
        #     all_label.append(13)

        # 'n/a': 27'
        # elif i == 'n/a':
        #     all_label.append(5)

        else:
            all_label.append(-1)

    all_label = np.array(all_label)

    data = []
    label = []

    for i in range(all_data.shape[0]):
        if all_label[i] != -1:
            data.append(all_data[i])
            label.append(all_label[i])

    data = np.array(data)
    label = np.array(label)

    n_sample = data.shape[0]

    split_idx_1 = int(0.75 * n_sample)
    split_idx_2 = int(0.85 * n_sample)

    shuffle_idx = np.random.permutation(n_sample)
    data = data[shuffle_idx]
    label = label[shuffle_idx]

    X_train = data[:split_idx_1]
    X_val = data[split_idx_1:split_idx_2]
    X_test = data[split_idx_2:]
    Y_train = label[:split_idx_1]
    Y_val = label[split_idx_1:split_idx_2]
    Y_test = label[split_idx_2:]

    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []

    for i in range(X_train.shape[0]):
        tmp_ts = X_train[i]
        tmp_Y = Y_train[i]
        ###############################################
        # 0: Myocardial infarction (368)

        # 1: 'Healthy control': 80

        # 2: 'Dysrhythmia': 16

        # 3: 'Cardiomyopathy': 17

        # 4: 'Bundle branch block': 17

        # 5: 'n/a': 27'
        ####################################
        if tmp_Y == 0:
            i_stride = 1
        elif tmp_Y == 1:
            i_stride = 4
        elif tmp_Y == 2:
            i_stride = 30
        elif tmp_Y == 3:
            i_stride = 18
        elif tmp_Y == 4:
            i_stride = 17
        else:
            i_stride = 21
        # if tmp_Y == 0:
        #     i_stride = 1
        # else:
        #     i_stride = 4

        for i in range(i_stride):
            train_data.append(tmp_ts)
            train_label.append(tmp_Y)

    for i in range(X_val.shape[0]):
        tmp_ts = X_val[i]
        tmp_Y = Y_val[i]
        # ###############################################
        # # 0: Myocardial infarction (368)
        #
        # # 1: 'Healthy control': 80
        #
        # # 2: 'Dysrhythmia': 16
        #
        # # 3: 'Cardiomyopathy': 17
        #
        # # 4: 'Bundle branch block': 17
        #
        # # 5: 'n/a': 27'
        # ####################################
        # if tmp_Y == 0:
        #     i_stride = 1
        # elif tmp_Y == 1:
        #     i_stride = 4
        # elif tmp_Y == 2:
        #     i_stride = 30
        # elif tmp_Y == 3:
        #     i_stride = 18
        # elif tmp_Y == 4:
        #     i_stride = 17
        # else:
        #     i_stride = 21
        # # if tmp_Y == 0:
        # #     i_stride = 1
        # # else:
        # #     i_stride = 4
        i_stride = 1
        for i in range(i_stride):
            val_data.append(tmp_ts)
            val_label.append(tmp_Y)

    for i in range(X_test.shape[0]):
        tmp_ts = X_test[i]
        tmp_Y = Y_test[i]
        # ###############################################
        # # 0: Myocardial infarction (368)
        #
        # # 1: 'Healthy control': 80
        #
        # # 2: 'Dysrhythmia': 16
        #
        # # 3: 'Cardiomyopathy': 17
        #
        # # 4: 'Bundle branch block': 17
        #
        # # 5: 'n/a': 27'
        # ####################################
        # if tmp_Y == 0:
        #     i_stride = 1
        # elif tmp_Y == 1:
        #     i_stride = 4
        # elif tmp_Y == 2:
        #     i_stride = 30
        # elif tmp_Y == 3:
        #     i_stride = 18
        # elif tmp_Y == 4:
        #     i_stride = 17
        # else:
        #     i_stride = 21
        # # if tmp_Y == 0:
        # #     i_stride = 1
        # # else:
        # #     i_stride = 4
        i_stride = 1
        for i in range(i_stride):
            test_data.append(tmp_ts)
            test_label.append(tmp_Y)

    X_train = np.array(train_data)
    Y_train = np.array(train_label)
    X_val = np.array(val_data)
    Y_val = np.array(val_label)
    X_test = np.array(test_data)
    Y_test = np.array(test_label)

    # shuffle train
    shuffle_pid = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def readmts_ptb_aug(filename):
    ## Loading time serie signals
    files = sorted(glob.glob(filename + "*.mat"))
    count = 0
    maxlen = 0

    all_data = []
    for i in range(len(files)):
        # if i in index2del:
        #     continue
        f = files[i]
        record = f[:-4]
        record = record[-6:]
        # Loading
        mat_data = scipy.io.loadmat(f[:-4] + ".mat")
        print('Loading record {}'.format(record))
        data = mat_data['val'].squeeze()
        all_data.append(data)
        data = np.array(data)
        if len(data) > maxlen:
            maxlen = len(data)
    all_data = np.array(all_data)

    # # normalize
    # for i in range(len(all_data)):
    #     tmp_data = all_data[i]
    #     tmp_std = np.std(tmp_data)
    #     tmp_mean = np.mean(tmp_data)
    #     all_data[i] = (tmp_data - tmp_mean) / tmp_std

    ## read label
    label_df = pd.read_csv(os.path.join(filename, 'REFERENCE-v3.csv'), header=None)
    label = label_df.iloc[:, 1].values
    all_label = []
    for i in label:

        # Myocardial infarction (368)
        if i == 'Myocardial infarction':
            all_label.append(0)

        # 'Healthy control': 80
        elif i == 'Healthy control':
            all_label.append(1)

        # 'Valvular heart disease': 6
        # elif i == 'Valvular heart disease':
        #     all_label.append(2)

        # 'Dysrhythmia': 16
        elif i == 'Dysrhythmia':
            all_label.append(2)

        # 'Heart failure (NYHA 2)': 1
        # elif i == 'Heart failure (NYHA 2)':
        #     all_label.append(4)

        # 'Heart failure (NYHA 3)': 1
        # elif i == 'Heart failure (NYHA 3)':
        #     all_label.append(5)

        # 'Heart failure (NYHA 4)': 1
        # elif i == 'Heart failure (NYHA 4)':
        #     all_label.append(6)

        # 'Palpitation': 1
        # elif i == 'Palpitation':
        #     all_label.append(7)

        # 'Cardiomyopathy': 17
        elif i == 'Cardiomyopathy':
            all_label.append(3)

        # 'Stable angina': 2
        # elif i == 'Stable angina':
        #     all_label.append(9)

        # 'Hypertrophy': 7
        # elif i == 'Hypertrophy':
        #     all_label.append(10)

        # 'Bundle branch block': 17
        elif i == 'Bundle branch block':
            all_label.append(4)

        # 'Unstable angina': 1
        # elif i == 'Unstable angina':
        #     all_label.append(12)

        # 'Myocarditis': 4
        # elif i == 'Myocarditis':
        #     all_label.append(13)

        # 'n/a': 27'
        # elif i == 'n/a':
        #     all_label.append(5)
        else:
            all_label.append(-1)

    all_label = np.array(all_label)

    data = []
    label = []

    for i in range(len(all_label)):
        if all_label[i] != -1:
            data.append(all_data[i])
            label.append(all_label[i])

    data = np.array(data)
    label = np.array(label)

    n_sample = len(label)

    split_idx_1 = int(0.75 * n_sample)
    split_idx_2 = int(0.85 * n_sample)

    flag = 1
    while flag == 1:
        shuffle_idx = np.random.permutation(n_sample)
        data = data[shuffle_idx]
        label = label[shuffle_idx]

        X_train = data[:split_idx_1]
        X_val = data[split_idx_1:split_idx_2]
        X_test = data[split_idx_2:]
        Y_train = label[:split_idx_1]
        Y_val = label[split_idx_1:split_idx_2]
        Y_test = label[split_idx_2:]

        train_labels = len(np.unique(Y_train, axis=0))
        val_labels = len(np.unique(Y_val, axis=0))
        test_labels = len(np.unique(Y_test, axis=0))
        if train_labels == val_labels and train_labels == test_labels:
            flag = 0

    # slide and cut
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    s1 = Counter(Y_train)[0] / Counter(Y_train)[1]
    s2 = Counter(Y_train)[0] / Counter(Y_train)[2] + 10
    s3 = Counter(Y_train)[0] / Counter(Y_train)[3]
    s4 = Counter(Y_train)[0] / Counter(Y_train)[4]
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=5000, stride=500, s1=s1, s2=s2, s3=s3, s4=s4,
                                     output_pid=False)

    s1 = Counter(Y_val)[0] / Counter(Y_val)[1]
    s2 = Counter(Y_val)[0] / Counter(Y_val)[2] + 5
    s3 = Counter(Y_val)[0] / Counter(Y_val)[3]
    s4 = Counter(Y_val)[0] / Counter(Y_val)[4]
    X_val, Y_val = slide_and_cut(X_val, Y_val, window_size=5000, stride=500, s1=s1, s2=s2, s3=s3, s4=s4,
                                 output_pid=False)

    s1 = Counter(Y_test)[0] / Counter(Y_test)[1]
    s2 = Counter(Y_test)[0] / Counter(Y_test)[2] + 10
    s3 = Counter(Y_test)[0] / Counter(Y_test)[3]
    s4 = Counter(Y_test)[0] / Counter(Y_test)[4]
    X_test, Y_test = slide_and_cut(X_test, Y_test, window_size=5000, stride=500, s1=s1, s2=s2, s3=s3, s4=s4,
                                   output_pid=False)

    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    print(X_train.shape, X_val.shape, X_test.shape)

    # shuffle train
    shuffle_pid = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def slide_and_cut(X, Y, window_size, stride, s1, s2, s3, s4, output_pid=False):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]

        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            i_stride = int(stride // s1)  # 4.6
        elif tmp_Y == 2:
            i_stride = int(stride // s2)  # 23
        elif tmp_Y == 3:
            i_stride = int(stride // s3)  # 21
        elif tmp_Y == 4:
            i_stride = int(stride // s4)  # 21

        # for j in range(0, len(tmp_ts)-window_size, i_stride):
        #     out_X.append(tmp_ts[j:j+window_size])
        #     out_Y.append(tmp_Y)
        #     out_pid.append(i)
        for j in range(0, len(tmp_ts) - window_size, i_stride):
            X_tmp = tmp_ts[j:j + window_size]
            out_X.append(X_tmp)
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)


def transform_labels(y_train, y_test, y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None:
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train, y_val, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val, new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test


def calculate_metrics(y_true, y_pred):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['Accuracy', 'Precision(macro)', 'Recall(macro)', 'Duration'])
    res['Accuracy'] = accuracy_score(y_true, y_pred)
    res['Precision(macro)'] = precision_score(y_true, y_pred, average='macro')  # sklearn.metrics 中直接计算precision
    res['Recall(macro)'] = recall_score(y_true, y_pred, average='macro')  # sklearn.metrics 中直接计算recall
    res['F1(macro)'] = f1_score(y_true, y_pred, average='macro')
    res['Precision(micro)'] = precision_score(y_true, y_pred, average='micro')
    res['Recall(micro)'] = recall_score(y_true, y_pred, average='micro')
    res['F1(micro)'] = f1_score(y_true, y_pred, average='micro')
    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')


def plot_trainingsize_metric(data, file_name):
    plt.figure()
    plt.plot(data["training_size"], data["accuracy"])
    plt.plot(data["training_size"], data["f1"])
    plt.title('model ')
    plt.ylabel('metric', fontsize='large')
    plt.xlabel('training_size', fontsize='large')
    plt.legend(['accuracy', 'f1'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close('all')


def plot_confusion_matrix(cm, nb_classes, output_directory, title='Confusion Matrix', cmap=plt.cm.binary):
    tick_marks = np.array(range(nb_classes))

    labels = []
    for i in range(nb_classes):
        labels.append("Cls." + str(i))

    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 12), dpi=120)

    ind_array = np.arange(nb_classes)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=100 / nb_classes, va='center', ha='center')
    # offset the tick
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # xlocations = np.array(range(nb_class))
    # plt.xticks(xlocations, xlocations, rotation=90)
    # plt.yticks(xlocations, xlocations)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_directory, format='png')
    plt.close('all')


def plot_confusion_matrix_seaborn(cm, nb_class, output_directory, title='Confusion Matrix'):
    labels = []
    for i in range(nb_class):
        labels.append(("Cls." + str(i)))
    sns.set(font_scale=1.5)
    f, hm = plt.subplots(figsize=(25, 25))
    hm = sns.heatmap(cm,
                     xticklabels=True,
                     yticklabels=True,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='d',
                     annot_kws={'size': 20})
    hm.set_xticklabels(labels, fontsize=18, horizontalalignment='right')
    hm.set_yticklabels(labels, fontsize=18, horizontalalignment='right')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cc.png', format='png')
    plt.savefig(output_directory, format='png')
    plt.close('all')


def plot_metrics_matrix(Precision, Recall, F1, nb_classes, output_directory):
    metrics = np.zeros((3, nb_classes))
    for i in range(3):
        for j in range(nb_classes):
            if i == 0:
                metrics[i, j] = Precision[j, 0]
            elif i == 1:
                metrics[i, j] = Recall[j, 0]
            else:
                metrics[i, j] = F1[j, 0]
    col_labels = []
    for i in range(nb_classes):
        col_labels.append("Cls." + str(i))
    row_labels = ['Precision', 'Recall', 'F1-score']
    sns.set(font_scale=2.5)
    f, hm = plt.subplots(figsize=(25, 25))
    hm = sns.heatmap(metrics,
                     xticklabels=True,
                     yticklabels=True,
                     cmap="YlGnBu",
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 20})
    hm.set_xticklabels(col_labels, fontsize=18, horizontalalignment='right')
    hm.set_yticklabels(row_labels, fontsize=18, horizontalalignment='right')
    plt.title('Metrics')
    plt.savefig(output_directory, format='png')
    plt.close('all')


def save_training_logs(output_directory, hist, lr=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)
    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 12), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss',
                                          'best_model_train_acc', 'best_model_val_acc',
                                          'best_model_train_precision', 'best_model_val_precision',
                                          'best_model_train_recall', 'best_model_val_recall',
                                          'best_model_train_f1', 'best_model_val_f1',
                                          'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    df_best_model['best_model_train_precision'] = row_best_model['precision']
    df_best_model['best_model_val_precision'] = row_best_model['val_precision']
    df_best_model['best_model_train_recall'] = row_best_model['recall']
    df_best_model['best_model_val_recall'] = row_best_model['val_recall']
    df_best_model['best_model_train_f1'] = row_best_model['f1']
    df_best_model['best_model_val_f1'] = row_best_model['val_f1']

    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
    plot_epochs_metric(hist, output_directory + 'epochs_accuracy.png', metric='accuracy')
    plot_epochs_metric(hist, output_directory + 'epochs_precision.png', metric='precision')
    plot_epochs_metric(hist, output_directory + 'epochs_recall.png', metric='recall')
    plot_epochs_metric(hist, output_directory + 'epochs_f1.png', metric='f1')

    return df_best_model


def save_evaluating_result(output_directory, y_pred, y_true, nb_classes):
    df_metrics = calculate_metrics(y_true, y_pred)

    cvconfusion = confusion_matrix(y_true, y_pred)

    # cvconfusion = np.zeros((nb_classes,nb_classes))
    # cvconfusion = confusion_matrix(y_true,y_pred)
    # cvconfusion_val = np.zeros((nb_classes,nb_classes))
    # cvconfusion_val = confusion_matrix(y_true,y_pred)

    classes = list(np.array(range(nb_classes)))

    df_confusion = pd.DataFrame(data=cvconfusion, index=classes, columns=classes)

    df_confusion.to_csv(output_directory + 'confusion_matrix.csv')

    plot_confusion_matrix(cvconfusion, nb_classes, output_directory + 'confusion_matrix.png')

    # print("hhhh")

    plot_confusion_matrix_seaborn(cvconfusion, nb_classes, output_directory + 'confusion_matrix(2).png')

    F1 = np.zeros((nb_classes, 1))
    Precision = np.zeros((nb_classes, 1))
    Recall = np.zeros((nb_classes, 1))

    for i in range(nb_classes):
        F1[i, 0] = 2 * cvconfusion[i, i] / (
                np.sum(cvconfusion[i, :]) + np.sum(cvconfusion[:, i]))
        Precision[i, 0] = cvconfusion[i, i] / np.sum(cvconfusion[:, i])
        Recall[i, 0] = cvconfusion[i, i] / np.sum(cvconfusion[i, :])

    df_metrics['F1(all)'] = np.mean(F1[0:nb_classes])

    for i in range(nb_classes):
        df_metrics['Precison(Cla.' + str(i) + ')'] = Precision[i, 0]
        df_metrics['Recall(Cla.' + str(i) + ')'] = Recall[i, 0]
        df_metrics['F1(Cla.' + str(i) + ')'] = F1[i, 0]

    plot_metrics_matrix(Precision, Recall, F1, nb_classes, output_directory + 'metrics.png')

    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    return cvconfusion, df_metrics
