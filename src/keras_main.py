import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras_model import Keras
from tensorflow.keras.utils import to_categorical
import utility as util
from preprocessing import Preprocessing


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
    dataset = Preprocessing(path)
    dataset.get_normalized_data()
    dataset.reshape_data()
    return dataset


def greeting_message():
    print("The program includes 5 configurations that reflects different settings the Keras model.")
    print("The Keras model has the similar structure in all configurations.")
    print("The differences are only in additional regularization parameters.")
    print()

    print("The 1 configuration does not include any regularization parameters.")
    print("The 2 configuration includes L1 regularization parameter.")
    print("The 3 configuration includes L2 regularization parameter.")
    print("The 4 configuration includes dropout regularization parameter.")
    print("The 5 configuration includes L2 and dropout regularization parameters.")
    print()

    print("It is necessary to choose a configuration of the model.")
    print()


def choose_configuration():
    L1 = 0.01
    L2 = 0.01
    configuration = int(input("Configuration [1, 2, 3, 4, 5, 6 (final model)]: "))
    if configuration == 2:
        L1 = float(input("it needs to choose the L1 value? [0.01, 0.001]: "))
    if configuration == 3:
        L2 = float(input("it needs to choose the L2 value? [0.01, 0.001]: "))
    if configuration == 5:
        L2 = float(input("it needs to choose the L2 value? [0.01, 0.001]: "))
    if configuration == 6:
        L2 = 0.001
        print("\nSince you\'ve chosen the final model. Could you allow me to explain what it means.")
        print("The final model reflets the most accuracy and the smallest loss based on passed dataset.\n")
        print("The final configuration based on 5 configuration that includes the droptout and L2 regularization parameters") 
    print()
    return (configuration, L1, L2)


def run_neuralnetwork(config, train, valid, cat_train_l, cat_valid_l, L1=0.01, L2=0.01):
    NN = Keras(config, L1=L1, L2=L2, dropout=0.5, verbose=1)
    compile_and_train(NN, train, cat_train_l, valid, cat_valid_l)
    return NN


def prepare_report_files(config=0, evaluate=None, report=None, conf_matrix=None, L1 = None, L2 = None):

    curr_dir = os.getcwd()
    output_path = os.path.join(curr_dir, "Outputs")

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    if os.path.exists(output_path):
        os.chdir(output_path)

    curr_dir = os.getcwd()

    config_path = os.path.join(curr_dir, "Congiguration_" + (str(config)) if config != 6 else "Final_configuration")
    if not os.path.exists(config_path) and config != 0:
        os.mkdir(config_path)
        
    if os.path.exists(config_path):
        os.chdir(config_path)

    file_1 = 'result.txt'

    configuration = ''

    match config:
        case 1:
            configuration = "Configuration {}: without regularization".format(config)
        case 2:
            configuration = "Configuration {}: with regularizer L1 = {}".format(config, L1)
        case 3:
            configuration = "Configuration {}: with regularizer L2 = {}".format(config, L2)
        case 4:
            configuration = "Configuration {}: with regularizer dropout = {}".format(config, 0.5)
        case 5:
            configuration = "Configuration {}: with regularizer L2 = {} and dropout = {}".format(config, L2, 0.5)
        case 6:
            configuration = "Final configuration: with regularizer L2 = {} and dropout = {}".format(L2, 0.5)

    with open(file_1, 'w') as result:
        result.write(configuration + '\n' + '\n')

        if evaluate != None:
            result.write('Accuracy = {:.2%}\n'.format(evaluate[1]))
            result.write('Loss = {:.4f}\n\n'.format(evaluate[0]))


        if report != None:
            result.write(report)

    if conf_matrix.any():
        df = pd.DataFrame(conf_matrix)
        df.to_csv('confusion_matrix')


def print_evaluate(train, value='loss'):
    plt.plot(train.history['accuracy'])
    plt.plot(train.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
 
    
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
    tst = tst.astype('float32')

    # Change the labels from integer to categorical data
    cat_train_l = to_categorical(train_l)
    cat_valid_l = to_categorical(valid_l)
    cat_tst_l = to_categorical(tst_l)

    greeting_message()
    (config, L1, L2) = choose_configuration()

    NN = None
    if config != 6:
        NN = run_neuralnetwork(config, train, valid, cat_train_l, cat_valid_l, L1, L2)
    else:
        NN = run_neuralnetwork(5, train, tst, cat_train_l, cat_tst_l, L1, L2)

    evaluate = None
    if config != 6:
        evaluate = NN.get_evaluate(valid, cat_valid_l)
    else:
        evaluate = NN.get_evaluate(tst, cat_tst_l)

    train = NN.get_train()

    print_evaluate(train, 'loss')
    print_evaluate(train, 'accuracy')

    #NN.model_info()
    
    NN.predict(valid)
    report = NN.get_report(cat_valid_l)

    confusion_matrix = NN.get_confusion_matrix(cat_valid_l)

    prepare_report_files(config, evaluate, report, confusion_matrix, L1, L2)


if __name__ == '__main__':
    main()

