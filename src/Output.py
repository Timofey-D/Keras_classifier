
import pandas as pd
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
