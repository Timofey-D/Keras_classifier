import sys
import numpy as np
from matplotlib import pyplot as plt
from keras_model import Keras
from tensorflow.keras.utils import to_categorical
import utility as util
from preprocessing import Preprocessing as Prep


def data_info(TR, TS, TRL, TSL):
    print("INFORMATION BY DATASET")

    print('Training data:', TR.shape, TRL.shape)
    print('Testing data:', TS.shape, TSL.shape)
    
    len_labels = len(np.unique(TRL))

    print('Labels:', np.unique(TRL))
    print('Total labels:', len_labels)


def run_config(NN, config, train, CTRL, test, CTSL, l_value=0.01, dropout=0.5, batch_size=256):
    if config == 1:
        NN.configuration()
    elif config == 2:
        NN.set_L1(l_value)
        NN.configuration_L1()
        print("With regularization L1 = {}".format(NN.get_L1()))
    elif config == 3:
        NN.set_L2(l_value)
        NN.configuration_L2()
        print("With regularization L2 = {}".format(NN.get_L2()))
    elif config == 4:
        NN.configuration_dropout()
        NN.set_dropout(dropout)
        print("With dropout dropout = {}".format(NN.get_dropout()))
    elif config == 5:
        NN.configuration_L2_dropout()
        NN.set_dropout(dropout)
        NN.set_L2(l_value)
        print("With dropout and regularization L2 = {}, dropout = {}".format(NN.get_L2(), NN.get_dropout()))

    NN.compilation()
    NN.train(train, CTRL, test, CTSL, iteration=50, batch=batch_size)
    NN.print_evaluate(test, CTSL)


def main():
    path_to_dataset = util.get_path(sys.argv)

    print(path_to_dataset)

    dataset = Prep(path_to_dataset)
    dataset.get_normalized_data()
    dataset.reshape_data()
    data = dataset.get_data()
    labels = dataset.get_labels()

    (TR, VAT, TRL, VATL) = dataset.split_data(data, labels, size_of_test=0.2, rand_state=None)
    (V, TS, VL, TSL) = dataset.split_data(VAT, VATL, size_of_test=0.5, rand_state=None)

    data_info(
            TR, VAT,
            TRL, VL
    )

    ## Change each value of the array to float
    TR = TR.astype('float32')
    V = V.astype('float32')

    # Change the labels from integer to categorical data
    CTRL = to_categorical(TRL)
    CVL = to_categorical(VL)

    NN = Keras()

    configurations = [1, 2, 3, 4, 5]

    run_config(NN, 1, TR, CTRL, V, CVL)

    NN.predict(V)
    conf_matrix = NN.get_confusion_matrix(CVL)
    print(conf_matrix)

    report = NN.get_report(CVL)
    print(report)


    #for config in configurations:

    #    print("Configuration {}:".format(config))

    #    if config == 2 or config == 3 or config == 5:
    #        lambdas = [0.01, 0.001]
    #        for l_value in lambdas:
    #            run_config(config, TR, CTRL, V, CVL, l_value, 0.5)
    #    else:
    #        run_config(config, TR, CTRL, V, CVL)

    #    conf_matrix = NN.get_confusion_matrix(test, CTSL)
    #    print(conf_matrix)

if __name__ == '__main__':
    main()
