import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from dlvc.models.cnn import CNN

# for wandb users:
from dlvc.wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''
        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''
        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''
        pass

class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5) -> None:
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)
            wandb_logger (WandBLogger): logs metrics during training to WeightsAndBiases account

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''

        ## TODO implement
        self.model = model 
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs 
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        
        # DataLoaders for Map-style datasets
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

        self.wandb_logger = WandBLogger(model=self.model)
        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        epoch_loss, epoch_acc, epoch_pc_acc = 0.0, 0.0, 0.0
        
        for idx, (inputs, targets) in enumerate(self.train_loader): # batch_idx (Tensor, Tensor)
            inputs, targets = inputs.to(self.device), targets.to(self.device) #move data to device gpu/cpu
            self.optimizer.zero_grad() #clear gradients from previous iteration
            outputs = self.model(inputs) # output for restnet [batch_size, 1000], should be [batch_size, 10] for cifar10
            loss = self.loss_fn(outputs, targets) # calculate loss
            loss.backward() # calculate gradients
            self.optimizer.step() # update weights | MODEL LEARNS HERE
            epoch_loss += loss.item()
            # Update training metric
            self.train_metric.update(outputs, targets)
            
        epoch_acc = self.train_metric.accuracy()
        epoch_pc_acc = self.train_metric.per_class_accuracy()
        print(f"Training Epoch: {epoch_idx} Loss: {loss}")
        print(self.train_metric)

        return epoch_loss, epoch_acc, epoch_pc_acc


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        self.model.eval()
        epoch_loss, epoch_acc, epoch_pc_acc = 0.0, 0.0, 0.0
        for idx, (inputs, targets) in enumerate(self.val_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            epoch_loss += loss.item()
            self.val_metric.update(outputs, targets)

        epoch_acc = self.train_metric.accuracy()
        epoch_pc_acc = self.train_metric.per_class_accuracy()
        print(f"Validation Epoch: {epoch_idx} Loss: {loss}")
        print(self.train_metric)

        return epoch_loss, epoch_acc, epoch_pc_acc
        
    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ## TODO implement
        curr_val_epoch_pc_acc = 0.0
        keys = [
            "train/loss",
            "train/mAcc",
            "train/mClassAcc",
            "val/loss",
            "val/mAcc",
            "val/mClassAcc"
        ]
        epoch_dict = dict.fromkeys(keys, 0.0)

        for epoch_idx in range(1, self.num_epochs+1):
            epoch_loss, epoch_acc, epoch_pc_acc = self._train_epoch(epoch_idx)

            if epoch_idx % self.val_frequency == 0:
                val_epoch_loss, val_epoch_acc, val_epoch_pc_acc = self._val_epoch(epoch_idx)

                epoch_dict.update({
                "val/loss": val_epoch_loss,
                "val/mAcc": val_epoch_acc,
                "val/mClassAcc": val_epoch_pc_acc
                })
                
                if val_epoch_pc_acc > curr_val_epoch_pc_acc:
                    self.model.save(self.training_save_dir)
                    curr_val_epoch_pc_acc = val_epoch_pc_acc

            epoch_dict.update({
                "train/loss": epoch_loss,
                "train/mAcc": epoch_acc,
                "train/mClassAcc": epoch_pc_acc
                })
            self.wandb_logger.log(log_dict=epoch_dict, step=epoch_idx)
        self.wandb_logger.finish()

                
# Tests
# def main(DATA_PATH = "cifar-10-batches-py"):
#     from torchvision.models import resnet18
#     NUM_CLASSES = 10
#     NUM_EPOCHS = 1

#     train_transform = v2.Compose([v2.ToImage(), 
#                             v2.ToDtype(torch.float32, scale=True),
#                             v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
#     val_transform = v2.Compose([v2.ToImage(), 
#                             v2.ToDtype(torch.float32, scale=True),
#                             v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    
#     train_data = CIFAR10Dataset(DATA_PATH, subset = Subset.TRAINING, transform=train_transform)
    

#     val_data = CIFAR10Dataset(DATA_PATH, subset = Subset.VALIDATION, transform=val_transform)
            
#     device = torch.device("cpu")
        
    
#     model_ft = resnet18(pretrained=False)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES) # 10 classes convertion
    
#     cnn_model = model_ft
#     cnn_model = CNN()
#     model = DeepClassifier(cnn_model)
#     model.to(device)
    
#     optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
#     loss_fn = torch.nn.CrossEntropyLoss()
    
#     train_metric = Accuracy(classes=train_data.classes)
#     val_metric = Accuracy(classes=val_data.classes)
#     val_frequency = 1

#     lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
#     model_save_dir = Path("saved_models")
#     model_save_dir.mkdir(exist_ok=True)
#     trainer = ImgClassificationTrainer(model, 
#                     optimizer,
#                     loss_fn,
#                     lr_scheduler,
#                     train_metric,
#                     val_metric,
#                     train_data,
#                     val_data,
#                     device,
#                     NUM_EPOCHS, 
#                     model_save_dir,
#                     batch_size=128, # feel free to change
#                     val_frequency = val_frequency)
#     # trainer._train_epoch(1)
#     # trainer._val_epoch(1)
#     trainer.train()


# if __name__ == "__main__":
#     import argparse
#     import os
#     import torch
#     import torchvision.transforms.v2 as v2
#     from pathlib import Path
#     import os
#     import torch.nn as nn
#     from dlvc.models.class_model import DeepClassifier # etc. change to your model
#     from dlvc.metrics import Accuracy
#     from dlvc.trainer import ImgClassificationTrainer
#     from dlvc.datasets.cifar10 import CIFAR10Dataset
#     from dlvc.datasets.dataset import Subset
#     from torch.optim import AdamW
#     from torch.optim.lr_scheduler import ExponentialLR
#     # files_and_directories = os.listdir("cifar-10-batches-py")#./cifar-10-batches-py
#     # print(files_and_directories)
#     main()


            
            


