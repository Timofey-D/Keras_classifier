import os
import sys


def get_path(input):
    path = 'Unknown'

    try:
        # To get a path to dataset
        path = os.path.join(os.getcwd(), sys.argv[1])
    except:
        raise Exception("The directory does not exist or the command was entered wrong!")

    return path
