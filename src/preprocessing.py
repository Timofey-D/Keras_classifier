import os
import cv2
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


class Preprocessing:

    def __init__(self, datadir, height=32, width=32):

        self.__check_datadir__(datadir)
        self.dataset = load_files(self.path)
        self.filenames = self.dataset.filenames
        self.labels = np.array(self.dataset.target)
        self.__normalized_data__(height, width)


    def __check_datadir__(self, datadir):
        path = None

        try:
            # To get a path to dataset
            path = os.path.join(os.getcwd(), datadir)
        except:
            raise Exception("The directory does not exist or the command was entered wrong!")

        self.path = path


    # To get a pad to image
    def __add_padding__(self, image):
        
        (h, w) = image.shape[:2]

        l_side = (h if h > w else w) / 2

        if h > w:
            segment = int(w + l_side)
            image = cv2.copyMakeBorder(image, 0, 0, segment, segment, cv2.BORDER_REPLICATE)
        else:
            segment = int(h + l_side)
            image = cv2.copyMakeBorder(image, segment, segment, 0, 0, cv2.BORDER_REPLICATE)

        return image

    # To split a dataset on several groups
    def split_data(self, data, labels, size_of_test=0.2, rand_state=6, shuffling=True):
        (X, Y, LX, LY) = train_test_split(data, labels, test_size=size_of_test, random_state=rand_state, shuffle=shuffling)
        return (X, Y, LX, LY)

    def reshape_data(self, width=32, height=32):
        # To reshape data
        self.data = self.data.reshape((self.data.shape[0], width*height))

    def get_dataset(self):
        return self.dataset

    
    def get_normalized_images(self):
        return self.normalized_images


    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    # To get a normalized image
    def __normalize_image__(self, image, height=32, width=32):
        # To add padding
        p_image = self.__add_padding__(image) 

        # To get an image size
        (h, w) = p_image.shape[:2]
        m_side = min(h, w)

        # To center an image
        c_image = p_image[
                int(h / 2 - m_side / 2) : int(h / 2 + m_side / 2), 
                int(w / 2 - m_side / 2) : int(w / 2 + m_side / 2)
                ]

        # To change an image size
        r_image = cv2.resize(c_image, (height,width), cv2.INTER_AREA)

        return r_image

    # To open an image by passed path
    def open_image(self, image_path):

        try:
            # To open an image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except:
            print("The path:", image_path)
            raise Exception("The file wasn\'t found or the file doesn't exist!")

        return image

    # To print out an image
    def print_image(self, image):
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # To get a normalized data
    def __normalized_data__(self, h=32, w=32):
        self.normalized_images = []
        
        for path in self.filenames:
            raw_image = self.open_image(path)
            image = self.__normalize_image__(raw_image, h, w)
            image = 255 - image
            self.normalized_images.append(image)

        self.data = np.array(self.normalized_images)

