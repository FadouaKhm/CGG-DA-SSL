# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:36:36 2022

@author: user
"""
import os
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.autograd import Variable
import configparser
import logging
def make_weight_for_balanced_classes(images, nclasses):
    count = [0]*nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.]*nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0]*len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def get_flir(dataset_root, batch_size, train):
    """Get FLIR datasets loader
    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set
    Returns:
        obj: dataloader object for FLIR dataset
    """ 
    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5776, 0.5776, 0.5776),
                                          std=(0.1319, 0.1319, 0.1319))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'sgada_data/flir/train'),
                                             transform=pre_process)
        weight = make_weight_for_balanced_classes(flir_dataset.imgs, len(flir_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5587, 0.5587, 0.558),
                                          std=(0.1394, 0.1394, 0.1394))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'sgada_data/flir/val'),
                                            transform=pre_process)
        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return flir_data_loader
datasetDir = r'C:\Users\user\Desktop\FLIR code'
target_train_loader = get_flir(datasetDir, 64, train=True)
target_val_loader = get_flir(datasetDir, 64, train=False)
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
transform = T.Compose([
     T.RandomResizedCrop(64),
     T.RandomHorizontalFlip(),
     T.ToTensor(),
   ])
trainDataset = ImageFolder('./data/flir/train', transform=transform)
trainDataLoader = DataLoader(trainDataset, 
        batch_size=64, shuffle=True)
#The picture in cat folder corresponds to label 0 and dog corresponds to 1
print(trainDataset.class_to_idx)

#Paths of all pictures and corresponding labels
print(trainDataset.imgs)

#There is no transform, so the PIL image object is returned
#Print (dataset [0] [1]) # the first dimension is the number of images, the second dimension is 1, and label is returned
#Print (dataset [0] [0]) # is 0 and returns picture data
plt.imshow(trainDataset[1][0])
plt.axis('off')
plt.show()

def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure("{} batch".format(dataset_type),
		figsize=(10, 10))
	# loop over the batch size
	for i in range(0, 64):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]
		# show the image along with the label
		plt.imshow(image)
		plt.title(label)
		plt.axis("off")
	# show the plot
	plt.tight_layout()
	plt.show()
    

trainBatch = next(iter(trainDataLoader))
# valBatch = next(iter(valDataLoader))
# visualize the training and validation set batches
print("[INFO] visualizing training and validation batch...")
visualize_batch(trainBatch, trainDataset.classes, "train")
# visualize_batch(valBatch, valDataset.classes, "val")