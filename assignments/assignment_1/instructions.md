# Deep Learning for Visual Computing - Assignment 1

The first assignment allows you to become familiar with basic pytorch usage and covers deep learning for image classification. In `requirements.txt` you can find a list of packages used for a reference implementations. However, you do not have to use those (except for `torchvision` and `pytorch`). 

This text or the reference code might not be without errors. If you find a significant error (not just typos but errors that affect the assignment), please contact us via [email](mailto:dlvc@cvl.tuwien.ac.at). Students who find and report such errors will get extra points.

## Part 1 - Dataset

Familiarize yourself with the code in the code folder (assignment_1_code) and make sure that the file structure is preserved. Read the code and make sure that you understand what the individual classes and methods are doing.

[Download](https://www.cs.toronto.edu/~kriz/cifar.html) and extract the *Python* version of the CIFAR10 dataset somewhere *outside* the code folder. Read the website to understand which classes there are and how the data are structured and read.

Then implement the `CIFAR10Dataset` class (`datasets/cifar10.py`). Some parts of this assignment will be tested automatically so make sure to follow the instructions in the code exactly (note, for example the specified data types). Make sure the following applies. If not, you made a mistake somewhere:

* Number of samples in the individual datasets: 40000 (training), 10000 (validation), 10000 (test).
* Total number samples: 6000 per class
* Image shape: `(32, 32, 3)`, image type: `uint8` (before applying transforms like v2.ToImage(), v2.ToDtype(torch.float32, scale=True))
* Labels of first 10 training samples: `[6, 9, 9, 4, 1, 1, 2, 7, 8, 3]`
* Make sure that the color channels are in RGB order by displaying the images and verifying the colors are correct (e.g. with `matpltolib.pyplot.imsave`). The first 8 training images are `frog truck truck deer car car bird horse` and should look like this (not normalized): ![image info](img/train_imgs.png)

Do not modify, add or delete any other files.

## Part 2 - Metrics
Implement the `Accuracy` class (`metrics.py`), which will help us with keeping track of the metrics accuracy and per-class accuracy. Read the doc strings for more detailed instructions and make sure to implement the specified type checks.

## Part 3 - Models
In the folder `dlvc/models` you will find 3 files, where you have to implement something in each file.
1.  In the file `class_model.py` you can find our wrapper-model `DeepClassifier`, which is wich stores the actual model in `self.net`. Implement the `save` and `load` method which are used for saving the best model during training (in the training loop in `dlvc/trainer.py`) and for testing (in the `test.py` file), where the saved model has to be loaded again. Look at the pytorch documentation for [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) and how to use `state_dict`.
2. In the file `cnn.py` define a PyTorch CNN with an architecture suitable for Image classification. This can be a simple model with a few layers. (PyTorch (and other libraries) expects the channel dimension of a single sample to be the first one, rows the second one, and columns the third one, `CHW` for short. You should already have this format given the things implemented above.)
3. In the file `vit.py` implement the vision transformer for classification. Here you can use existing repositories or blog posts. The goal is to be able to use existing code and modify it so that it can be integrated with the existing code base. However, you will need to be able to understand what is happening within the code you are using (see report part).
4. No coding necessary for this: Pytorch, in particular torchvision, comes with a range of [Models](https://pytorch.org/vision/stable/models.html). You should import the resnet18 model (without pretrained weights thus simply `from torchvision.models import resnet18` and `resnet18()` gives you an instance of this model) and use it for one of your experiments (see part 5).

## Part 4 - Training loop 
Implement the `ImgClassificationTrainer` class (`dlvc/trainer.py`). This class holds the entire training logic. See the doc strings in the class methods for additional information.
1. Implement the method `train` which holds the outer training loop, the loop over the number of epochs. Keep track of the train and validation loss and metrics using the `Accuracy` class implemented in Part 2 of this exercise, mean accuracy and mean per-class accuracy). You can either use [Weights & Biases](https://wandb.ai/site) (free for academic purposes), tensorboard or store metrics in a list and plot them afterwards. (You will have to include those plots in the report.) Save the model if mean per-class accuracy on the validation dataset is higher than the currently saved best mean per-class accuracy (and overwrite that file in the process of training). This way you keep the model that performed best on the validation set for later for testing.
2. Implement the `_train_epoch` and `_val_epoch` methods, that hold the logic for training or validating one epoch. In there you need to loop over the batches in the corresponding dataloaders. Print the results after every epoch, similar to this:
```python
______epoch 0

accuracy: 0.4847
per class accuracy: 0.4848
Accuracy for class: plane is 0.52 
Accuracy for class: car   is 0.58 
Accuracy for class: bird  is 0.32 
Accuracy for class: cat   is 0.32 
Accuracy for class: deer  is 0.39 
Accuracy for class: dog   is 0.42 
Accuracy for class: frog  is 0.59 
Accuracy for class: horse is 0.54 
Accuracy for class: ship  is 0.63 
Accuracy for class: truck is 0.53 
```
Note: make sure your data (and model) are always on the correct device. Make use of `model.eval()`, `model.train()` and `with torch.no_grad()` etc. when appropriate.


## Part 5 - Putting it all together
In the file `train.py` everything is put together and the training is started. In there you specify the model, loss, optimizer, learning rate scheduler, image transformations, and hyper-parameters such as number of epochs and batch size. Further, the device used is setup, either cpu or gpu. The training is started with `Trainer(...).train()`. Make sure that your script is runable!

In the file `test.py` you need to implement the testing setup: creating test dataset, test dataloader, test loop (similar to validation loop), loading the trained model etc. 

You should run 3 models: Your Pytorch CNN implementation, the resnet18 from `torchvision.models` and your ViT implementation. Put your train and validation metric curves (loss, accuracy and per class accuracy) as figures in the report and submit the best configurations you tried (optimizer, learning rate, lr scheduler, epochs etc. - you might need to use different lr or num epochs per model. However you do not need to do an extensive hyperparameter search, if you see the model is far from it expected capacity, try a different lr or morr epochs etc. but do not get lost in this; most important part is to be able to run all three and get okayish results.) for each model in separate training files, `train_yourCNN.py`, `train_yourViT.py`, `train_resnet18.py`.

For reference, using the resnet18 model, the given transforms, a batchsize of 128, validation frequency of 5, 30 epochs, ExponentialLR as lr scheduler with gamma = 0.9, AdamW with amsgrad=True and lr=0.001 as optimizer and CrossEntropyLoss we get the following results on the test set,
```python
test loss: 1.6786944755554198

accuracy: 0.7508
per class accuracy: 0.7508
Accuracy for class: plane is 0.81 
Accuracy for class: car   is 0.86 
Accuracy for class: bird  is 0.69 
Accuracy for class: cat   is 0.55 
Accuracy for class: deer  is 0.73 
Accuracy for class: dog   is 0.61 
Accuracy for class: frog  is 0.82 
Accuracy for class: horse is 0.80 
Accuracy for class: ship  is 0.84 
Accuracy for class: truck is 0.80 
```
and our training and validation curves look like this:

![alt text](img/train_val_curves.png)

Most likely as you can also see in the plots above you will encounter some overfitting. To address this pick one model (resnet18, your CNN, or your ViT) and experiment with different strategies discussed, namely data augmentation and regularization:

* Data augmentation: You are allowed to used torchvision.transforms.v2 functions to apply them on the training set. Have a look at the documentation. Try at least random crops and left/right mirroring. 
* Regularization: use at least weight decay but you can also experiment with dropout (in addition to or instead of weight decay).


In the report you should discuss how the individual techniques affected your training and validation set accuracy and performance on the test set. Don't just compare part 2 and part 3 results, also compare at least a few combinations and settings like only regularization but no data augmentation vs. both, different regularization strengths and so on. **This may take some time, so don't delay working on the assignment until shortly before the deadline.** Try a couple combinations (at least 2).

To get additional points you experiment with different learning rate schedulers, and optimizers. Discuss your findings in the report.


## Report

Write a short report (3 to 4 pages) that includes your findings from your experiments and answers the following questions:
* What is the purpose of a loss function? What does the cross-entropy measure? Which criteria must the ground-truth labels and predicted class-scores fulfill to support the
cross-entropy loss, and how is this ensured?
* What is the purpose of the training, validation, and test sets and why do we need all of them?
* What are the goals of data augmentation, regularization, and early stopping? How exactly did you use these techniques (hyperparameters, combinations) and what were your results (train, val and test performance)? List all experiments and results, even if they did not work well, and discuss them.
* What are the key differences between ViTs and CNNs? What are the underlying conceptes, respectively? Give the source of your ViT model implementation and point to the specific lines of codes (in your code) where the key concept(s) of the ViT are and explain them.

Also include your results obtained from `train.py` and `test.py`. Include the validation and train (per-class) accuracies as a **plot** as well as the final test (per-class) accuracy. Compare the best validation accuracy and the final test accuracy, and discuss the results. Furthermore, state which optimizer, learning rate scheduler etc. (and their parameteres) were used.

## Submission

Submit your assignment until **April 30th at 11pm**. To do so, create a zip archive including the report, the complete `dlvc` folder with your implementations (do not include the CIFAR-10 dataset), the `test.py`, `train_yourCNN.py`, `train_yourViT.py`, `train_resnet18.py` files . More precisely, after extracting the archive we should obtain the following:

    group_x/
        report.pdf
        train_yourCNN.py
        train_yourViT.py
        train_resnet18.py
        test.py
        dlvc/
            metrics.py
            ...
            datasets/
                ...
            ...

Submit the zip archive on TUWEL. Make sure you've read the general assignment information [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss23/-/blob/main/assignments/general.md) before your final submission.


## Server Usage

You may find that training is slow on your computer unless you have an Nvidia GPU with CUDA support. If so, copy the code into your home directory on the DLVC server and run it there. The login credentials will be sent out on April 28th - check your spam folder if you didn't. For details on how to run your scripts see [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss23/-/blob/main/assignments/DLVC2023Guide.pdf). For technical problems regarding our server please contact [email](mailto:dlvc-trouble@cvl.tuwien.ac.at).

We expect queues will fill up close to assignment deadlines. In this case, you might have to wait a long time before your script even starts. In order to minimize wait times, please do the following:

* Write and test your code locally on your system. If you have a decent Nvidia GPU, please train locally and don't use the servers. If you don't have such a GPU, perform training for a few epochs on the CPU to ensure that your code works. If this is the case, upload your code to our server and do a full training run there. To facilitate this process, have a variable or a runtime argument in your script that controls whether CUDA should be used. Disable this locally and enable it on the server.
* Don't schedule multiple training runs in a single job, and don't submit multiple long jobs. Be fair.
* If you want to train on the server, do so as early as possible. If everyone starts two days before the deadline, there will be long queues and your job might not finish soon enough.
