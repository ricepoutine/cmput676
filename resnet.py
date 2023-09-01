### Reference: https://github.com/bentrevett/pytorch-image-classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import numpy as np
from pathlib import Path

import copy
from collections import namedtuple
import os
import random
import shutil
import time

from plot import normalize_image, plot_images, count_parameters, plot_lr_finder, plot_confusion_matrix, plot_filtered_images, plot_filters
from helper import get_features


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h
    
    
class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x
    
    
class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        
        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr = 10, num_iter = 100, 
                   smooth_f = 0.05, diverge_th = 5):
        
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        
        iterator = IteratorWrapper(iterator)
        
        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            #update lr
            lr_scheduler.step()
            
            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
                
            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            
            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
                       
        #reset model to initial parameters
        self.model.load_state_dict(torch.load('init_params.pt'))
                    
        return lrs, losses

    def _train_batch(self, iterator):
        
        self.model.train()
        
        self.optimizer.zero_grad()
        
        x, y = iterator.get_batch()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_pred, _ = self.model(x)
                
        loss = self.criterion(y_pred, y)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()



class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]



class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)


def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k


def train(model, iterator, optimizer, criterion, scheduler, device, k=5):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_k = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred, _ = model(x)
        
        loss = criterion(y_pred, y)
        
        acc_1, acc_k = calculate_topk_accuracy(y_pred, y, k=k)
        
        loss.backward()
        
        optimizer.step()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_k += acc_k.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_k /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_k


def evaluate(model, iterator, criterion, device, k=5, extract_features = False, fdict = {}, flist = [], plist = []):
    
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_k = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)
            
            if extract_features:
                #add feats and preds to list
                y_prob = F.softmax(y_pred, dim = -1)
                top_pred = y_prob.argmax(1, keepdim = True)
                plist.append(top_pred.detach().cpu().numpy())
                flist.append(fdict['feats'].cpu().numpy())

            loss = criterion(y_pred, y)

            acc_1, acc_k = calculate_topk_accuracy(y_pred, y, k=k)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_k += acc_k.item()
        
    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_k /= len(iterator)
        
    return epoch_loss, epoch_acc_1, epoch_acc_k


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_predictions(model, iterator, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def main():
    
    # We'll set the random seeds for reproducability.
    SEED = 42
    BATCH_SIZE = 64 #initially 200
    CPUS = 8
    EPOCHS = 4
    N_IMAGES = 5
    N_FILTERS = 7

    learn_means_from_data = False #set to false and load dict to use pretrained
    show_sample_images = False
    print_model = False
    find_learning_rate = False

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    
    # image_dir = Path("/Volumes/Data/Work/Research/2022_10_ResNet/images")
    image_dir = Path.cwd() / "images"
    train_dir = image_dir / "train"
    test_dir = image_dir / "test"

    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]

    train_data = datasets.ImageFolder(root = train_dir, transform = transforms.ToTensor())

    if learn_means_from_data:
        means = torch.zeros(3)
        stds = torch.zeros(3)

        for img, label in train_data:
            means += torch.mean(img, dim = (1,2))
            stds += torch.std(img, dim = (1,2))

        means /= len(train_data)
        stds /= len(train_data)
        print(f'Calculated means: {means}')
        print(f'Calculated stds: {stds}')
    else:
        # these values are from the pretrained ResNet on 1000-class imagenet data
        means = [0.485, 0.456, 0.406]
        stds= [0.229, 0.224, 0.225]
    
    pretrained_size = 224
    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std = stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.CenterCrop(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std = stds)
                        ])
    
    # We load our data with our transforms...
    train_data = datasets.ImageFolder(root = train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(root = test_dir, transform = test_transforms)

    VALID_RATIO = 0.25

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
    
    # ...and then overwrite the validation transforms, making sure to 
    # do a `deepcopy` to stop this also changing the training data transforms.
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms
    
    # To make sure nothing has messed up we'll print the number of examples 
    # in each of the data splits - ensuring they add up to the number of examples
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    
    # Next, we'll create the iterators with the largest batch size that fits on our GPU. 
    train_iterator = data.DataLoader(train_data, 
                        shuffle=True, 
                        batch_size=BATCH_SIZE, 
                        num_workers=CPUS, 
                        persistent_workers=True)
    valid_iterator = data.DataLoader(valid_data, 
                        batch_size=BATCH_SIZE, 
                        num_workers=CPUS, 
                        persistent_workers=True)
    test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=CPUS)
    
    print()
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of training iterations:   {len(train_iterator)}")
    print(f"Number of validation iterations: {len(valid_iterator)}")
    print(f"Number of test iterations:       {len(test_iterator)}")
    
    # To ensure the images have been processed correctly we can plot a few of them - 
    # ensuring we re-normalize the images so their colors look right.
    if show_sample_images:
        N_IMAGES = 25

        images, labels = zip(*[(image, label) for image, label in 
                                [train_data[i] for i in range(N_IMAGES)]])

        classes = test_data.classes
        plot_images(images, labels, classes)
    
    # We will use a `namedtuple`to store: 
    #   the block class, 
    #   the number of blocks in each layer, 
    #   and the number of channels in each layer.
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2, 2, 2, 2],
                               channels = [64, 128, 256, 512])
    
    resnet34_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

    # Below are the configurations for the ResNet50, ResNet101 and ResNet152 models. 
    # Similar to the ResNet18 and ResNet34 models, the `channels` do not change between configurations, 
    # just the number of blocks in each layer.
    resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

    resnet101_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 23, 3],
                                    channels = [64, 128, 256, 512])

    resnet152_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 8, 36, 3],
                                    channels = [64, 128, 256, 512])
    
    # The images in our dataset are 768x768 pixels in size. 
    # This means it's appropriate for us to use one of the standard ResNet models.
    # We'll choose ResNet50 as it seems to be the most commonly used ResNet variant. 

    # As we have a relatively small dataset - with a very small amount of examples per class - 40 images - 
    # we'll be using a pre-trained model.

    # Torchvision provides pre-trained models for all of the standard ResNet variants
    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # We can see that the final linear layer for the classification, `fc`, has a 1000-dimensional 
    # output as it was pre-trained on the ImageNet dataset, which has 1000 classes.
    if print_model:
        print(pretrained_model)
    
    # Our dataset, however, only has 2 classes, so we first create a new linear layer with the required dimensions.
    IN_FEATURES = pretrained_model.fc.in_features 
    OUTPUT_DIM = len(test_data.classes)
    fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
    
    # Then, we replace the pre-trained model's linear layer with our own, randomly initialized linear layer.
    # **Note:** even if our dataset had 1000 classes, the same as ImageNet, we would still remove the 
    # linear layer and replace it with a randomly initialized one as our classes are not equal to those of ImageNet.
    pretrained_model.fc = fc
    
    # The pre-trained ResNet model provided by torchvision does not provide an intermediate output, 
    # which we'd like to potentially use for analysis. We solve this by initializing our own ResNet50 
    # model and then copying the pre-trained parameters into our model.

    # We then initialize our ResNet50 model from the configuration...
    model = ResNet(resnet50_config, OUTPUT_DIM)
    
    # ...then we load the parameters (called `state_dict` in PyTorch) of the pre-trained model into our model.
    # This is also a good sanity check to ensure our ResNet model matches those used by torchvision.
    
    ###TESTING NO PRETRAINING
    model.load_state_dict(pretrained_model.state_dict())
    
    # We can also see the number of parameters in our model - noticing that ResNet50 only has ~24M parameters 
    # compared to VGG11's ~129M. This is mostly due to the lack of high dimensional linear layers which 
    # have been replaced by more parameter efficient convolutional layers.
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # filters = model.conv1.weight.data
    # plot_filters(filters, title="Before")
    
    # ### Training the Model
    #
    # Next we'll move on to training our model. As in previous notebooks, we'll use the learning rate finder to 
    # set a suitable learning rate for our model.

    # We start by initializing an optimizer with a very low learning rate, defining a loss function (`criterion`) 
    # and device, and then placing the model and the loss function on to the device.
    START_LR = 1e-7
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    device = torch.device('mps') #mps for mac
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Extract features: https://github.com/kozodoi/website/blob/master/_notebooks/2021-05-27-extracting-features.ipynb

    # placeholders
    PREDS = []
    FEATS = []

    # placeholder for batch features
    features = {}

    # print(model)
    model.avgpool.register_forward_hook(get_features('feats', dict=features))

    # We then define our learning rate finder and run the range test.
    if find_learning_rate:
        END_LR = 10
        NUM_ITER = 100
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)
        
        # We can see that the loss reaches a minimum at around $3x10^{-3}$.
        # A good learning rate to choose here would be the middle of the steepest downward curve - which is around $1x10^{-3}$.
        plot_lr_finder(lrs, losses, skip_start = 30, skip_end = 30)
    
    # We can then set the learning rates of our model using discriminative fine-tuning - a technique 
    # used in transfer learning where later layers in a model have higher learning rates than earlier ones.

    # We use the learning rate found by the learning rate finder as the maximum learning rate - used in the final layer - 
    # whilst the remaining layers have a lower learning rate, gradually decreasing towards the input.
    
    FOUND_LR = 1e-3
    params = [
            {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
            {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
            {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
            {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
            {'params': model.fc.parameters()}
            ]

    optimizer = optim.Adam(params, lr = FOUND_LR)
    
    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
    
    print(f"Starting training...")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS} | Steps/Epoch: {STEPS_PER_EPOCH} | Total Steps: {TOTAL_STEPS}")

    MAX_LRS = [p['lr'] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr = MAX_LRS,
                                        total_steps = TOTAL_STEPS)
    
    # Finally, we can train our model!
    best_valid_loss = float('inf')

    # remove if already trained!!!

    for epoch in range(EPOCHS):
        
        start_time = time.monotonic()
        
        train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, scheduler, device, k=1)
        valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device, k=1, extract_features = False)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut5-model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
            f'Train Acc @1: {train_acc_5*100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
            f'Valid Acc @1: {valid_acc_5*100:6.2f}%')

    ### TODO: take best model features and generate data points to fit gaussians

    model.load_state_dict(torch.load('tut5-model.pt'))
    _, _, _ = evaluate(model, valid_iterator, criterion, device, k=1, extract_features = True, fdict=features, flist=FEATS, plist=PREDS)

    # Inspect features (TEST)

    PREDS = np.concatenate(PREDS)
    FEATS = np.resize(np.concatenate(FEATS), (n_valid_examples,2048))

    print('- preds shape:', PREDS.shape)
    print('- feats shape:', FEATS.shape)

    import csv
    with open('PREDS_test.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(PREDS)
        f.close()
    with open('FEATS_test.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(FEATS)
        f.close()
    
    '''
    # Examine the test accuracies
    model.load_state_dict(torch.load('tut5-model.pt'))

    test_loss, test_acc_1, test_acc_k = evaluate(model, test_iterator, criterion, device, k=1, extract_features = False, flist = FEATS, plist = PREDS)

    print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | ' \
        f'Test Acc @1: {test_acc_k*100:6.2f}%')
    
    # ### Examining the Model
    # Get the predictions for each image in the test set...
    print()
    print("Getting predictions for images in the test set...")
    images, labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)
    
    # Plot the confusion matrix for the test results
    plot_confusion_matrix(labels, pred_labels, classes)
    
    # Show several images after they have been through the 'conv1' convolutional layer
    filters = model.conv1.weight.data
    il = [(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]]
    images, labels = zip(*il)
    plot_filtered_images(images, labels, classes, filters, n_filters=N_FILTERS)
    
    plot_filters(filters, title='After')
    '''
    return
        
        
        
if __name__ == "__main__":
    main()
