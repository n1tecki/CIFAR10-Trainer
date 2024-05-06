from pathlib import Path
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from dlvc.models.vit import YourViT
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from dlvc.models.class_model import DeepClassifier


def train(args):
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((32, 32)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((32, 32)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CIFAR10Dataset(args.data_path, subset=Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(args.data_path, subset=Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vit = YourViT()
    model = DeepClassifier(model_vit)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 1

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

    trainer = ImgClassificationTrainer(model, optimizer, loss_fn, lr_scheduler, train_metric,
                                       val_metric, train_data, val_data, device, args.num_epochs,
                                       model_save_dir, batch_size=128, val_frequency=val_frequency)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training setup for ViT model')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of epochs to train for')
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU ID to use')
    parser.add_argument('--data_path', required=True, type=str, help='Directory path to CIFAR-10 dataset')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    train(args)



