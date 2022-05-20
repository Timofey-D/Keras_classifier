import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mode import Mode


def greeting():
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

    
def main():
    greeting()
    mode = int(input("Enter the program mode [1, 2, 3, 4, 5, 6 (final)]: "))

    program = Mode(mode, sys.argv[1], height=32, width=32, train_size=0.2, test_size=0.5)

    program.run_mode(256, 50, 0)
    NN = program.get_keras()
    report = NN.get_report()
    accuracy = report['accuracy']

    print(report['accuracy'], report['loss'], sep='\n')


if __name__ == '__main__':
    main()

