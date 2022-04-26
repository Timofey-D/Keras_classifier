from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow


class Keras:

    def __init__(self, config, L1=0.01, L2=0.01, dropout=0.5, verbose=0):

        self.NN = Sequential()

        self.L1 = L1
        self.L2 = L2
        self.dropout = dropout
    
        match config:
            case 1:       
                if verbose == 1:
                    print("Configuration {}: without regularizers".format(config))
                self.configuration()
            case 2:
                if verbose == 1:
                    print("Configuration {}: with regularizer L1 = {}".format(config, self.L1))
                self.configuration_L1()
            case 3:
                if verbose == 1:
                    print("Configuration {}: with regularizer L2 = {}".format(config, self.L2))
                self.configuration_L2()
            case 4:
                if verbose == 1:
                    print("Configuration {}: with regularizer dropout = {}".format(config, self.dropout))
                self.configuration_dropout()
            case 5:
                if verbose == 1:
                    print("Configuration {}: with regularizer L2 = {} and dropout = {}".format(config, self.L2, self.dropout))
                self.configuration_L2_dropout()


    def model_info(self):
        print(self.NN.summary())


    def configuration(self):

        self.NN.add(Dense(1024, activation='relu'))
        self.NN.add(Dense(512))
        self.NN.add(Dense(512))
        self.NN.add(Dense(27, activation='softmax'))


    def configuration_L1(self):
        
        self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L1(self.L1)))
        self.NN.add(Dense(512, kernel_regularizer=regularizers.L1(self.L1)))
        self.NN.add(Dense(512, kernel_regularizer=regularizers.L1(self.L1)))
        self.NN.add(Dense(27, activation='softmax'))


    def configuration_L2(self):

        self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(self.L2)))
        self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
        self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
        self.NN.add(Dense(27, activation='softmax'))


    def configuration_dropout(self):

        self.NN.add(Dense(1024, activation='relu'))
        self.NN.add(Dropout(self.dropout))
        self.NN.add(Dense(512))
        self.NN.add(Dropout(self.dropout))
        self.NN.add(Dense(512))
        self.NN.add(Dropout(self.dropout))
        self.NN.add(Dense(27, activation='softmax'))


    def configuration_L2_dropout(self):

        self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(self.L2)))
        self.NN.add(Dropout(self.dropout))
        self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
        self.NN.add(Dropout(self.dropout))
        self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
        self.NN.add(Dropout(self.dropout))
        self.NN.add(Dense(27, activation='softmax'))


    def train(self, train, cat_train_l, valid, cat_valid_l, batch=32, iteration=100, verb=1):
        self.train = self.NN.fit(
                train, 
                cat_train_l, 
                batch_size=batch, 
                epochs=iteration, 
                verbose=verb, 
                validation_data=(valid, cat_valid_l)
        )

    def set_L1(self, value):
        self.L1 = value

    def set_L2(self, value):
        self.L2 = value

    def set_dropout(self, value):
        self.dropout = value

    def print_evaluate(self, data, labels):
        [loss, accuracy] = self.NN.evaluate(data, labels)
        print("Evaluation result on Test Data : Loss = {}, accuracy = {:.2%}".format(loss, accuracy))

    def predict(self, data):
        self.prediction = self.NN.predict(data)

    def compilation(self):
        self.NN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        #self.NN.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    def get_L1(self):
        return self.L1

    def get_L2(self):
        return self.L2

    def get_dropout(self):
        return self.dropout

    def get_evaluate(self):
        evaluate = self.NN.evaluate(data, labels)
        return evalute

    def get_report(self, labels):
        report = classification_report(labels.argmax(axis=1), self.prediction.argmax(axis=1))
        return report

    def get_confusion_matrix(self, labels): 
        conf_matrix = confusion_matrix(labels.argmax(axis=1), self.prediction.argmax(axis=1))
        return conf_matrix

    def get_neural_network(self):
        return self.NN

    def get_train(self):
        return self.train

