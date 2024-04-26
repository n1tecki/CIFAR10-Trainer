from abc import ABCMeta, abstractmethod
from collections import defaultdict
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        self.correct_predictions = 0
        self.total_predictions = 0

        self.class_correct_predictions = defaultdict(int)
        self.class_correct_predictions.update({class_name: 0 for class_name in self.classes})
        
        self.class_total_predictions = defaultdict(int)
        self.class_total_predictions.update({class_name: 0 for class_name in self.classes})

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        ## TODO implement
        self.total_predictions += target.shape[0]
        predicted_classes = torch.argmax(prediction, dim=1)
        true_predictions = target[predicted_classes == target]

        for true_class in target:
            self.class_total_predictions[true_class] += 1

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        ## TODO implement
        pass


    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        ## TODO implement
        pass
    
    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        ## TODO implement
        pass
       