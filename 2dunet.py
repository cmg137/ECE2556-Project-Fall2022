import os,sys
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
#import torchvision.utils
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
#from loss import dice_loss
#from imageio import imread
from PIL import Image
# from sklearn.model_selection import KFold as kfold

# # Generate some random images
# input_images, target_masks = simulation.generate_random_data(192, 192, count=3)

#for x in [input_images, target_masks]:
   #  print(x.shape)
  #   print(x.min(), x.max())

# # Change channel-order and make 3 channels for matplot
# input_images_rgb = [x.astype(np.uint8) for x in input_images]

# # Map each channel (i.e. class) to each color
# target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]

# # Left: Input image, Right: Target mask (Ground-truth)
# #helper.plot_side_by_side([input_images_rgb, target_masks_rgb])


class SimDataset(Dataset):
    #Assemble dataset
    def __init__(self, test=False, transform=None):
        if test:
            self.input_images = os.listdir("label_testset")[-50:]
            self.target_masks = os.listdir("label_testset")[-50:]
        else:
            self.input_images = os.listdir("label_testset")
            self.target_masks = os.listdir("label_testset")

        self.transform = transform
    #return number of input_images
    def __len__(self):
        return len(self.input_images)
    #retreive an image and the mask for a given index
    def __getitem__(self, idx):        
        image = np.load("label_testset\\" + self.input_images[idx]).astype(np.uint8)
        image = Image.fromarray(image)
        mask = np.load("label_testset\\" + self.target_masks[idx][:10] + '.npy')
        mask = (255 * mask).astype(np.uint8)
        mask = Image.fromarray(mask)
        # mask = np.concatenate([[mask, mask, mask, mask, mask, mask]])
        # mask = (mask>0).astype(np.float32)
       # mask = mask.astype(np.uint8)

        if self.transform:
            image = self.transform(image)

            mask = self.transform(mask)


        return [image, mask]

# use same transform for train/val for this example
trans = transforms.Compose([
    transforms.Resize([1024, 1024]),
    transforms.ToTensor(),
])


train_set = SimDataset(transform = trans)
val_set = SimDataset(test=True, transform = trans)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 4

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

dataset_sizes

#function definition to perform ReLU convolution
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

#UNet structure
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        #pretrain model
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())


        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
    # forward propagation
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_class=1)
model = model.to(device)

summary(model, input_size=(3, 224, 224))


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    #REPLACED WITH TORCH.SIGMOID INSTEAD OF F.SIGMOID
    pred = torch.sigmoid(pred)
    #dice = dice_loss(pred, target)
    
    #loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = bce * bce_weight
    # loss = bce
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
   # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            total_acc = 0
            # correct = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)             
                correct = 0
                # zero the parameter gradients
                optimizer.zero_grad()
                #dupplicated input channels
                inputs = torch.cat([inputs,inputs,inputs],1)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    pred = torch.max(outputs.data, dim=1)[1]
                    pred = (pred > 0.).cpu().numpy().astype(np.float32)
                    labels = labels.cpu().numpy()
                    # correctness = 
                    correct = (pred == labels).mean()
                    print(correct)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if epoch % 10 == 0:
                print("saving")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), '/home/fxy/pytorch-unet/' + str(epoch) + '.pth')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




print(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, 500)