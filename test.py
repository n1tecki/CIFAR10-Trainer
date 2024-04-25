## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os

from torchvision.models import resnet18 # change to the model you want to test
from dlvc.models.class_model import DeepClassifier
from dlvc.metrics import Accuracy
from dlvc.datasets.cifar10 import CIFAR10Dataset
from dlvc.datasets.dataset import Subset
from torch.utils.data import DataLoader




def test(args):
   
    
    transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    
    test_data = CIFAR10Dataset(fdir=args.data_path, subset=Subset.TEST, transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_test_data = len(test_data)

    resnet18_model = resnet18(pretrained=False)
    model = DeepClassifier(resnet18_model)
    model.load(args.trained_model_path)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    test_metric = Accuracy(classes=test_data.classes)

    # Testing loop
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item() * data.size(0)
            total_accuracy += test_metric(output, target)

    avg_loss = total_loss / len(test_data)
    avg_accuracy = total_accuracy / len(test_data)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Testing')
    parser.add_argument('--gpu_id', default='0', type=str, help='Index of which GPU to use')
    parser.add_argument('--data_path', type=str, required=True, help='Directory path to CIFAR-10 dataset')
    parser.add_argument('--trained_model_path', type=str, required=True, help='Path to the trained model file')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    test(args)