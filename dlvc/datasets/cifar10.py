import pickle
from typing import Tuple
import numpy as np
import os
import sys
# sys.path.append('c:\\Users\\mariu\\Documents\\deep_learning_for_visual_computing\\dlvc_ss24') 


from dlvc.datasets.dataset import  Subset, ClassificationDataset

class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''
        self.transform = transform

        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.data = []
        self.labels = []
        

        if not os.path.isdir(fdir):
            raise ValueError("Directory does not exist!")
        
        if subset == Subset.TRAINING:
            batch_files = [f"data_batch_{i}" for i in range(1, 5)]
        elif subset == Subset.VALIDATION:
            batch_files = ["data_batch_5"]
        elif subset == Subset.TEST:
            batch_files = ["test_batch"]
        
        for file in batch_files:
            try:
                full_path = os.path.join(fdir, file)
                with open(full_path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
            except:
                raise ValueError(f"File {full_path} does not exist")
            
            self.labels.extend(dict[b'labels'])
            self.data.extend(dict[b'data'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.annot_labels = [self.classes[int(label)] for label in self.labels]

        
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)


    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        labels_array = np.array(self.labels)
        count_per_class = np.bincount(labels_array)
        return count_per_class



# Testing
# object = CIFAR10Dataset(fdir = "C:/Users/mariu/Documents/deep_learning_for_visual_computing/cifar-10-python/", subset = Subset.TRAINING)
# object.__len__()
# object.__len__()
# print(object.__getitem__(2))