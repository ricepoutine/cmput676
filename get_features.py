# generate feats and preds csvs
from collections import namedtuple
from resnet import evaluate, ResNet, Bottleneck, get_features
import torch
import torch.nn as nn
import random
import copy
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

device = torch.device('mps') #mps for mac

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

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)

n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

# ...and then overwrite the validation transforms, making sure to 
# do a `deepcopy` to stop this also changing the training data transforms.
valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

valid_iterator = data.DataLoader(valid_data, 
                    batch_size=BATCH_SIZE, 
                    num_workers=CPUS, 
                    persistent_workers=True)

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet50_config = ResNetConfig(block = Bottleneck,
                            n_blocks = [3, 4, 6, 3],
                            channels = [64, 128, 256, 512])
model = ResNet(resnet50_config, 2)

criterion = nn.CrossEntropyLoss()
model = model.to(device)

features = {}
PREDS = []
FEATS = []
model.avgpool.register_forward_hook(get_features('feats', dict=features))

model.load_state_dict(torch.load('tut5-model.pt'))
_, _, _ = evaluate(model, valid_iterator, criterion, device, k=1, extract_features = True, fdict=features, flist=FEATS, plist=PREDS)

# Inspect features (TEST)
PREDS = np.concatenate(PREDS)
FEATS = np.resize(np.concatenate(FEATS), (200,2048))

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