import os
import pandas as pd

from mode import Mode


class Output:
    """
        result.txt
        - configuration of the model
        - classification report
        - accuracy / loss
        confusion_matrix.csv
        - confusion matrix
    """
    def __init__(self, report, program_mode):

        self.report = report
        self.program = program_mode
        self.mode = self.program.get_mode()
        self.description = self.program.get_mode_info()
        self.directory = "Mode_" + str(self.mode)


    def create_report_directory(self):
        self.__mkdir__("Output")

        self.__mkdir__(self.directory)

        self.__create_result_file__()
        self.__create_confusion_matrix_file__()


    def __create_confusion_matrix_file__(self):

        df = pd.DataFrame(self.report['confusion matrix'])
        df.to_csv('confusion_matrix')


    def __create_result_file__(self):

        data = ['accuracy', 'loss', 'classification report']

        with open('result.txt', 'w') as f:
            f.write(self.description + '\n')
            f.write('\n')

            f.write(self.program.__data_info__())
            f.write('\n')

            L1 = self.program.get_keras().get_L1()
            L2 = self.program.get_keras().get_L2()
            dropout = self.program.get_keras().get_dropout()

            param = ''

            if L1 != None:
                param = f'{L1=}'.split('=')[0]
                value = f'{L1=}'.split('=')[1]
                f.write("{} = {}\n".format(param, value))

            if L2 != None:
                param = f'{L2=}'.split('=')[0]
                value = f'{L2=}'.split('=')[1]
                f.write("{} = {}\n".format(param, value))

            if dropout != None:
                param = f'{dropout=}'.split('=')[0]
                value = f'{dropout=}'.split('=')[1]
                param = param.capitalize()
                f.write("{} = {}\n".format(param, value))

            if param != '':
                f.write('\n')

            for param in data:

                line = str(self.report[param]) + '\n'

                if param == data[0] or param == data[1]:
                    line = "{}: {:.4}".format(param, self.report[param])

                f.write(str(line))
                f.write('\n')

            f.close()


    def __mkdir__(self, dirname):

        curr_dir = os.getcwd()
        path = os.path.join(curr_dir, dirname)

        if not os.path.exists(path):
            os.mkdir(path)

        os.chdir(path)


    '''
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

        df = pd.DataFrame(conf_matrix)
        df.to_csv('confusion_matrix.csv')
    '''
