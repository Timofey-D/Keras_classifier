from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import tensorflow


class Keras:

    L1 = None
    L2 = None
    dropout = None

    def __init__(self, train, valid, train_l, valid_l):

        self.NN = Sequential()

        # Change each value of the array to float
        self.train = train.astype('float32')
        self.valid = valid.astype('float32')

        # Change the labels from integer to categorical data
        self.train_l = to_categorical(train_l)
        self.valid_l = to_categorical(valid_l)
        
    
    def add_input_layer(self):

        if self.L1 != None:
            self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L1(self.L1)))
        elif self.L2 != None:
            self.NN.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(self.L2)))
        else:
            self.NN.add(Dense(1024, activation='relu'))


    def add_internal_layer(self):

        if self.L1 != None:
            self.NN.add(Dense(512, kernel_regularizer=regularizers.L1(self.L1)))
        elif self.L2 != None:
            self.NN.add(Dense(512, kernel_regularizer=regularizers.L2(self.L2)))
        else:
            self.NN.add(Dense(512))


    def add_output_layer(self):
        self.NN.add(Dense(27, activation='softmax'))


    def add_dropout_layer(self):
        self.NN.add(Dropout(self.dropout))


    def train_network(self, batch=32, iteration=100, verb=1):
        self.train = self.NN.fit(
                self.train, 
                self.train_l, 
                batch_size=batch, 
                epochs=iteration, 
                verbose=verb, 
                validation_data=(self.valid, self.valid_l)
        )

    
    def get_report(self):
        report = dict()

        prediction = self.NN.predict(self.valid)
        report.update( {'prediction' : prediction} )

        [loss, accuracy] = self.NN.evaluate(self.valid, self.valid_l)
        report.update( {'accuracy' : accuracy} )
        report.update( {'loss' : loss} )

        label_x1 = self.valid_l.argmax(axis=1)
        classification = classification_report(label_x1, prediction.argmax(axis=1))
        report.update( {'classification report' : classification} )

        conf_matrix = confusion_matrix(label_x1, prediction.argmax(axis=1))
        report.update( {'confusion matrix' : conf_matrix} )
        
        return report


    def info_model(self):
        self.NN.summary()

    def compilation(self):
        self.NN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def set_L1(self, value):
        self.L1 = value

    def set_L2(self, value):
        self.L2 = value

    def set_dropout(self, value):
        self.dropout = value

    def get_L1(self):
        return self.L1

    def get_L2(self):
        return self.L2

    def get_dropout(self):
        return self.dropout

    def get_NN(self):
        return self.NN

    def get_train(self):
        return self.train

    def print_evaluate(train, value='loss'):
        plt.plot(train.history['accuracy'])
        plt.plot(train.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

