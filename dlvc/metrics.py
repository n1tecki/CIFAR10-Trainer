from abc import ABCMeta, abstractmethod
from collections import defaultdict
import torch
import numpy as np
from typing import Tuple, Dict

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
    __slots__ = ["classes", "correct_predictions", "total_predictions", 
                 "class_correct_predictions", "class_total_predictions", "accuracy_per_class"]

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

        classes = self.classes
        if torch.is_tensor(classes):
            classes = classes.numpy()

        self.class_correct_predictions = defaultdict(int)
        self.class_correct_predictions.update({classes.index(class_name): 0 for class_name in classes})
        
        self.class_total_predictions = defaultdict(int)
        self.class_total_predictions.update({classes.index(class_name): 0 for class_name in classes})

        self.accuracy_per_class = defaultdict(int)
        self.class_total_predictions.update({classes.index(class_name): 0 for class_name in classes})

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        ## TODO implement
        # extracting numpy arrays from Tensor
        prediction = prediction.detach().numpy() #.detach() to create a new tensor that does not require gradients and then convert to numpy array
        target = target.detach().numpy()

        # updates for accuracy
        self.total_predictions += target.shape[0]
        predicted_classes = np.argmax(prediction, axis=1)
        true_predictions = target[predicted_classes == target]
        self.correct_predictions += true_predictions.shape[0]

        # updates dictionaries for class accuracy
        for true_class in target:
            self.class_total_predictions[true_class] += 1
        
        for true_prediction in true_predictions:
            self.class_correct_predictions[true_prediction] += 1


    def __str__(self) -> str:
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        ## TODO implement
        accuracy = self.accuracy()
        per_class_accuracy = self.per_class_accuracy()
        performance_str = f"Performance Metrics:\n"
        performance_str += f"Overall Accuracy: {accuracy:.2f}\n"
        performance_str += f"Average Per Class Accuracy: {per_class_accuracy:.2f}\n"
        for class_num, accuracy in self.accuracy_per_class.items():
            performance_str += f"{self.classes[class_num]} : {accuracy}\n"

        return performance_str

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        ## TODO implement
        if self.total_predictions == 0:
            return 0
        return self.correct_predictions / self.total_predictions

    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        ## TODO implement
        # calculates average per-class accuracy
        if self.total_predictions == 0:
            return 0
        
        self.accuracy_per_class = defaultdict(int)
        num_predicted_classes = 0
        for c, total_prediction in self.class_total_predictions.items():
            if total_prediction == 0:
                self.accuracy_per_class[c] = 0
            else:
                self.accuracy_per_class[c] = self.class_correct_predictions[c] / total_prediction
                num_predicted_classes += 1

        return sum(self.accuracy_per_class.values()) / num_predicted_classes

# Tests
# classes = torch.tensor(np.array([0,1,2,3]))
# classes = np.array([0,1,2,3])
# acc_obj = Accuracy(classes)
# prediction = torch.tensor(np.array([[0.1, 0.1, 0.1, 0.7],[0.35, 0, 0.4, 0.25],[0.4, 0.2, 0.3, 0.1]]))
# target = torch.tensor(np.array([3,0,0]))
# print(acc_obj)
# acc_obj.update(prediction, target)
# print(acc_obj.accuracy())
# print(acc_obj.per_class_accuracy())
# print(acc_obj)
# acc_obj.reset()
# print(acc_obj)


