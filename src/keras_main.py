import sys
import numpy as np
from matplotlib import pyplot as plt
from keras_model import Keras
from tensorflow.keras.utils import to_categorical
import utility as util
from preprocessing import Preprocessing as Prep


def data_info(train, test, train_l, test_l):
    print("INFORMATION BY DATASET")

    print('Training data:', train.shape, train_l.shape)
    print('Testing data:', test.shape, test_l.shape)
    
    np_labels = np.unique(train_l)
    len_labels = len(np_labels)

    print('Labels:', np_labels)
    print('Total labels:', len_labels)


def compile_and_train(NN, train, train_l, valid, valid_l):
    NN.compilation()
    NN.train(train, train_l, valid, valid_l, batch=256, iteration=50, verb=0)


def prepare_dataset(path):
    dataset = Prep(path)
    dataset.get_normalized_data()
    dataset.reshape_data()
    return dataset


def main():
    path_to_dataset = util.get_path(sys.argv)
    dataset = prepare_dataset(path_to_dataset)
    data = dataset.get_data()
    labels = dataset.get_labels()

    (train, test, train_l, test_l) = dataset.split_data(data, labels, size_of_test=0.2, rand_state=None)
    (valid, tst, valid_l, tst_l) = dataset.split_data(test, test_l, size_of_test=0.5, rand_state=None)

    print()
    data_info(train, test, train_l, test_l)
    print()
    data_info(valid, tst, valid_l, tst_l)
    print()

    # Change each value of the array to float
    train = train.astype('float32')
    valid = valid.astype('float32')

    # Change the labels from integer to categorical data
    cat_train_l = to_categorical(train_l)
    cat_valid_l = to_categorical(valid_l)

    configurations = [
            { 'config' : 1},
            { 'config' : 2, 'L1' : 0.01},
            { 'config' : 2, 'L1' : 0.001},
            { 'config' : 3, 'L2' : 0.01},
            { 'config' : 3, 'L2' : 0.001},
            { 'config' : 4, 'dropout' : 0.5},
            { 'config' : 5, 'L2' : 0.01, 'dropout' : 0.5},
            { 'config' : 5, 'L2' : 0.001, 'dropout' : 0.5}
    ]

    for config in configurations:
        NN = Keras(
                config=config['config'],
                L1=config['L1'] if list(config.keys()).count('L1') != 0 else 0.0,
                L2=config['L2'] if list(config.keys()).count('L2') != 0 else 0.0,
                dropout=config['dropout'] if list(config.keys()).count('dropout') != 0 else 0.0,
                verbose=1
        )

        compile_and_train(NN, train, cat_train_l, valid, cat_valid_l)
        NN.print_evaluate(valid, cat_valid_l)
        NN.model_info()

        NN.predict(valid)
        report = NN.get_report(cat_valid_l)
        print(report)

        confusion_matrix = NN.get_confusion_matrix(cat_valid_l)
        print(confusion_matrix)
        print()


if __name__ == '__main__':
    main()

