## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

import torch.nn as nn
from dlvc.models.class_model import DeepClassifier # etc. change to your model
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from torchvision.models import resnet18
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR



    
def train(args):
    
    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure 
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation

    train_transform = v2.Compose([
        v2.ToImage(), 
        v2.RandomCrop(32, padding=4),  # Augmentation
        v2.RandomHorizontalFlip(0.5),  # Augmentation
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    train_data = CIFAR10Dataset(args.data_path, subset = Subset.TRAINING, transform=train_transform)
    
    val_data = CIFAR10Dataset(args.data_path, subset = Subset.VALIDATION, transform=val_transform)
    
 
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ft = resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.layer4.register_forward_hook(lambda m, inp, out: nn.functional.dropout(out, p=0.5)) # Dropout in 4th layer
    model_ft.fc = nn.Linear(num_ftrs, 10) # 10 classes convertion
    model = DeepClassifier(model_ft)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True, weight_decay=0.01) # Weight decay
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    trainer = ImgClassificationTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=128, # feel free to change
                    val_frequency = val_frequency)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training setup')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of epochs to train for')
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU ID to use')
    parser.add_argument('--data_path', required=True, type=str, help='Directory path to CIFAR-10 dataset') 
    # "C:/Users/mariu/Documents/deep_learning_for_visual_computing/cifar-10-python/"
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    train(args)