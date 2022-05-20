from keras_model import Keras
from preprocessing import Preprocessing


class Mode:

    modes = [1, 2, 3, 4, 5, 6]

    def __init__(self, mode, datadir, height=32, width=32, train_size=0.2, test_size=0.5):

        self.__check_mode__(mode)
        self.mode = mode
        self.__data_preparation__(datadir, height, width, train_size, test_size)


    def run_mode(self, batch=32, iter_=100, verbose=0):

        self.keras = Keras(self.data_1, self.data_2, self.labels_1, self.labels_2)
        self.__build_model__()
        self.keras.compilation()
        self.keras.train_network(batch, iter_, verbose)


    def __build_model__(self):

        if self.mode == 1:
            self.__mode_1__()
        if self.mode == 2:
            self.keras.set_L1(float(input("Enter the L1 value: ")))
            self.__mode_1__()
        if self.mode == 3:
            self.keras.set_L2(float(input("Enter the L2 value: ")))
            self.__mode_1__()
        if self.mode == 4:
            self.keras.set_dropout(float(input("Enter the dropout value: ")))
            self.__mode_2__()
        if self.mode == 5:
            self.keras.set_L2(float(input("Enter the L2 value: ")))
            self.keras.set_dropout(float(input("Enter the dropout value: ")))
            self.__mode_3__()
        if self.mode == 6:
            self.keras.set_L2(0.001)
            self.keras.set_dropout(0.5)
            self.__mode_3__()


    def get_mode_info():
        pass


    def get_keras(self):
        return self.keras


    def get_list_modes(self):
        return self.modes


    def __check_mode__(self, mode):
        modes = self.get_list_modes()

        if modes.count(mode) == 0:
            raise Exception("The obtained mode does not exist!")


    def __data_preparation__(self, datadir, height, width, train_size=0.2, test_size=0.5):
        dataset = Preprocessing(datadir, height, width)
        dataset.reshape_data(height, width)
        data = dataset.get_data()
        labels = dataset.get_labels()

        (train, validating, train_l, validating_l) = dataset.split_data(data, labels, size_of_test=train_size, rand_state=None)
        (valid, test, valid_l, test_l) = dataset.split_data(validating, validating_l, size_of_test=test_size, rand_state=None)

        self.data_1 = train
        self.labels_1 = train_l

        if self.mode == 6:
            self.data_2 = test
            self.labels_2 = test_l
        else:
            self.data_2 = valid
            self.labels_2 = valid_l
        

    def data_info(train, test, train_l, test_l):
        print("INFORMATION BY DATASET")

        print('Training data:', train.shape, train_l.shape)
        print('Testing data:', test.shape, test_l.shape)
        
        np_labels = np.unique(train_l)
        len_labels = len(np_labels)

        print('Labels:', np_labels)
        print('Total labels:', len_labels)


    def __mode_1__(self):
        self.keras.add_input_layer()
        self.keras.add_internal_layer()
        self.keras.add_internal_layer()
        self.keras.add_output_layer()

    def __mode_2__(self):
        self.keras.add_input_layer()
        self.keras.add_dropout_layer()
        self.keras.add_internal_layer()
        self.keras.add_dropout_layer()
        self.keras.add_internal_layer()
        self.keras.add_dropout_layer()
        self.keras.add_output_layer()

    def __mode_3__(self):
        self.keras.add_input_layer()
        self.keras.add_dropout_layer()
        self.keras.add_internal_layer()
        self.keras.add_dropout_layer()
        self.keras.add_internal_layer()
        self.keras.add_dropout_layer()
        self.keras.add_output_layer()
'''
Dense(1024, activation='relu'))
Dense(512))
Dense(512))
Dense(27, activation='softmax'))
'''
''' 
self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L1(self.L1)))
self.NN.add(Dense(512, kernel_regularizer=regularizers.L1(self.L1)))
self.NN.add(Dense(512, kernel_regularizer=regularizers.L1(self.L1)))
self.NN.add(Dense(27, activation='softmax'))
'''
'''
self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(self.L2)))
self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
self.NN.add(Dense(27, activation='softmax'))
'''
'''
self.NN.add(Dense(1024, activation='relu'))
self.NN.add(Dropout(self.dropout))
self.NN.add(Dense(512))
self.NN.add(Dropout(self.dropout))
self.NN.add(Dense(512))
self.NN.add(Dropout(self.dropout))
self.NN.add(Dense(27, activation='softmax'))
'''
'''
self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(self.L2)))
self.NN.add(Dropout(self.dropout))
self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
self.NN.add(Dropout(self.dropout))
self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
self.NN.add(Dropout(self.dropout))
self.NN.add(Dense(27, activation='softmax'))
'''
