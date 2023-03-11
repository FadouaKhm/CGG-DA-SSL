import torch
from .data_utils import split_ssl_data, sample_labeled_data
from .dataset import BasicDataset
from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
import json
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import random
from .augmentation.randaugment import RandAugment
from sklearn.model_selection import train_test_split
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
from datasets.DistributedProxySampler import DistributedProxySampler
import h5py
import gc
import sys
import copy
from PIL import Image
from config import config

from torchvision import datasets, transforms
import torch.utils.data as data
from torch.autograd import Variable
import configparser
import logging

import numpy as np
from datasets.data_utils import ToTensorWithoutScaling




mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
mean['imagenet'] = [0.485, 0.456, 0.406]
mean['FLIR'] = [0.5776, 0.5776, 0.5776]
mean['ATR'] = [0, 0, 0]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['svhn'] = [0.1751, 0.1771, 0.1744]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
std['imagenet'] = [0.229, 0.224, 0.225]
std['FLIR'] = [0.1319, 0.1319, 0.1319]
std['ATR'] = [1,1,1]
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def map_y(y, from_dict, to_dict):
    inv_from = {v:k for k,v in from_dict.items()}
    y_names = [inv_from[e] for e in y]
    y_out = [to_dict[e] for e in y_names]
    return y_out
def select_classes(dataset):
    to_del = set(dataset['labels'])-set(config['input_dict'].values())
    masks = []
    for c in to_del:
        
        mask = np.where(dataset["labels"]==c)[0]
        masks.append(mask)
        dataset["labels"] = np.delete(dataset["labels"], mask,0)
        dataset["images"] = np.delete(dataset["images"], mask,0)
    dataset["labels"] = np.array(map_y(dataset["labels"], config['input_dict'], config['output_dict']))
    return dataset

def balance_data(input_data):
    idx = np.arange(len(input_data['labels']))
    # print(input_data['labels'].shape)
    output = dict()
    sel_idx = []
    #bal_num = min([Counter(list(input_data["labels"]))[k] for k in Counter(list(input_data["labels"])).keys()])
    bal_num = int(np.mean(list(Counter(input_data['labels']).values()))) 
    classes = list(Counter(list(input_data["labels"])).keys())

    for c in classes:
        mask = input_data['labels'] == c
        if output == {}:
            output['images'] = input_data['images'][mask,:,:,:][:bal_num,:,:,:]
            output['labels'] = input_data['labels'][mask][:bal_num]
            sel_idx = idx[mask][:bal_num]
        else:
            output['images'] = np.concatenate((output['images'],input_data['images'][mask,:,:,:][:bal_num,:,:,:]),axis=0)
            output['labels'] = np.concatenate((output['labels'],input_data['labels'][mask][:bal_num]),axis=0)
            sel_idx = np.concatenate((sel_idx,idx[mask][:bal_num]),axis=0) 
    # print (sel_idx.shape)
    # with open('sel_idx_test.npy', 'wb') as f:
    #     np.save(f, sel_idx)
    # f.close()
    return output
class ImagenetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ulb, num_labels=-1):
        super().__init__(root, transform)
        self.ulb = ulb
        self.num_labels = num_labels
        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.num_labels != -1:
                    fnames = fnames[:self.num_labels]
                if self.num_labels != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        if self.num_labels != -1:
            with open('./sampled_label_idx.json', 'w') as f:
                json.dump(lb_idx, f)
        del lb_idx
        gc.collect()
        return instances


class ImageNetLoader:
    def __init__(self, root_path, num_labels=-1, num_class=1000):
        self.root_path = os.path.join(root_path, 'imagenet')
        self.num_labels = num_labels // num_class

    def get_transform(self, train, ulb):
        if train:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        else:
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        return transform

    def get_lb_train_data(self):
        transform = self.get_transform(train=True, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=False,
                               num_labels=self.num_labels)
        return data

    def get_ulb_train_data(self):
        transform = self.get_transform(train=True, ulb=True)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=True)
        return data

    def get_lb_test_data(self):
        transform = self.get_transform(train=False, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "val"), transform=transform, ulb=False)
        return data


# def get_transform(mean, std, crop_size, train=True):
#     if train:
#         return transforms.Compose([transforms.RandomHorizontalFlip(),
#                                     transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
#                                     transforms.ToTensor()])
#     else:
#         return transforms.Compose([transforms.ToTensor()])
def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 args,
                 alg='fixmatch',
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='./data',
                 vae_data = False):
        """
        Args
            alg: SSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.args = args
        self.alg = alg
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.vae_data = vae_data
        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 64 if (self.name.upper() == 'ATR' or self.name.upper() == 'FLIR') else 32
        self.transform = get_transform(mean[name], std[name], crop_size, train)

    def get_data(self, svhn_extra=True):
        """
        get_data returns data (images) and targets (labels)
        shape of data: B, H, W, C
        shape of labels: B,
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        if 'CIFAR' in self.name.upper():
           # print(self.data_dir,self.train)
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
            return data, targets
        elif self.name.upper() == 'SVHN':
            if self.train:
                if svhn_extra:  # train+extra
                    dset_base = dset(self.data_dir, split='train', download=True)
                    data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
                    dset_extra = dset(self.data_dir, split='extra', download=True)
                    data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
                    data = np.concatenate([data_b, data_e])
                    targets = np.concatenate([targets_b, targets_e])
                    del data_b, data_e
                    del targets_b, targets_e
                else:  # train_only
                    dset = dset(self.data_dir, split='train', download=True)
                    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            else:  # test
                dset = dset(self.data_dir, split='test', download=True)
                data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            return data, targets
        elif self.name.upper() == 'STL10':
            split = 'train' if self.train else 'test'
            dset_lb = dset(self.data_dir, split=split, download=True)
            dset_ulb = dset(self.data_dir, split='unlabeled', download=True)
            data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
            ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
            return data, targets, ulb_data


    def get_atr_data(self,txt_data_dir, lb_files, ulb_files):
        dataset = {"images": [], "labels": []}
        unlabeled = {"images": [], "labels": []}

        for line in lb_files:
            file = h5py.File(txt_data_dir+'/{}.hdf5'.format(line), 'r')
            if len(dataset["images"]) == 0:
                dataset["images"] = file["images"][...]
                dataset["labels"] = file["labels"][...]
            else:
                dataset["images"] = np.concatenate([dataset["images"][...], file["images"][...]], axis=0)
                dataset["labels"] = np.concatenate([dataset["labels"][...], file["labels"][...]], axis=0)
            file.close()
        dataset = select_classes(dataset)
        dataset_ = balance_data(dataset)
        # dataset_ = dataset
        
        data, targets = dataset_["images"].transpose([0,2,3,1]), dataset_["labels"]
        data, targets = data.astype(np.uint8), targets.astype(np.int64)
        #X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
        
        if self.train and self.alg != 'fullysupervised':
            
            for line in ulb_files:
                file = h5py.File(txt_data_dir+'/{}.hdf5'.format(line), 'r')
                if len(unlabeled["images"]) == 0:
                    unlabeled["images"] = file["images"][...]
                    unlabeled["labels"] = file["labels"][...]
                else:
                    unlabeled["images"] = np.concatenate([unlabeled["images"][...], file["images"][...]], axis=0)
                    unlabeled["labels"] = np.concatenate([unlabeled["labels"][...], file["labels"][...]], axis=0)
                file.close()
            unlabeled["images"] = unlabeled["images"].transpose([0,2,3,1]).astype(np.uint8)
        return data, targets, unlabeled["images"]

    def get_flir_data(self,txt_data_dir, lb_files):
        dataset = {"images": [], "labels": []}

        for line in lb_files:
            file = h5py.File(txt_data_dir+'/{}.hdf5'.format(line), 'r')
            if len(dataset["images"]) == 0:
                dataset["images"] = file["images"][...]
                dataset["labels"] = file["labels"][...]
            else:
                dataset["images"] = np.concatenate([dataset["images"][...], file["images"][...]], axis=0)
                dataset["labels"] = np.concatenate([dataset["labels"][...], file["labels"][...]], axis=0)
            file.close()
        
        data, targets = dataset["images"].transpose([0,2,3,1]), dataset["labels"]
        data, targets = data.astype(np.uint8), targets.astype(np.int64)

        return data, targets
    # def get_flir(self, dataset_root):
    #     """Get FLIR datasets loader
    #     Args:
    #         dataset_root (str): path to the dataset folder
    #         batch_size (int): batch size
    #         train (bool): create loader for training or test set
    #     Returns:
    #         obj: dataloader object for FLIR dataset
    #     """ 
    #     if self.train:
    #         pre_process = transforms.Compose([transforms.Resize((64, 64)),ToTensorWithoutScaling()])
    #         flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'sgada_data/flir/train'),
    #                                              transform=pre_process)
    #         flir_loader = torch.utils.data.DataLoader(flir_dataset, batch_size=len(flir_dataset), num_workers=1)
    #         data, targets = next(iter(flir_loader))[0].numpy(), next(iter(flir_loader))[1].numpy()
    #         weight = make_weight_for_balanced_classes(flir_dataset.imgs, len(flir_dataset.classes))
    #         weight=torch.DoubleTensor(weight)
    
    #         sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    #         return np.transpose(data,(0,2,3,1)).astype(np.uint8), targets, sampler
    
    #     else:
    #         pre_process = transforms.Compose([transforms.Resize((64, 64)),ToTensorWithoutScaling()])
    #         flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'sgada_data/flir/val'),
    #                                             transform=pre_process)
    #         flir_loader = torch.utils.data.DataLoader(flir_dataset, batch_size=len(flir_dataset), num_workers=1)
    #         data, targets = next(iter(flir_loader))[0].numpy(), next(iter(flir_loader))[1].numpy()
    
    
    #         return np.transpose(data,(0,2,3,1)).astype(np.uint8), targets, None

    def get_dset(self, is_ulb=False,
                 strong_transform=None, onehot=False, txt_data_dir='', lb_files='', ulb_files='', dataset_root = ''):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            is_ulb: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is Trueã€‚
            onehot: If True, the label is not integer, but one-hot vector.
        """
        num_classes = self.num_classes
        transform = self.transform
        if self.name.upper() == 'STL10':
            data, targets, _ = self.get_data()
        elif self.name.upper() == 'ATR':
            data, targets, _= self.get_atr_data(txt_data_dir, lb_files, ulb_files)
        elif self.name.upper() == 'FLIR':
            data, targets= self.get_flir_data(txt_data_dir, lb_files)             
        else:
            data, targets = self.get_data()

        # print("isTrain =", self.train, "Shape: ",data.shape)
        # print("Classes: ",Counter(targets))
        return BasicDataset(self.alg, data, targets, num_classes, transform,
                                is_ulb, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     strong_transform=None, onehot=False, txt_data_dir='', lb_files='', ulb_files=''):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        # Supervised top line using all data as labeled data.
        if self.alg == 'fullysupervised' and (self.name.upper() != 'ATR'):
            lb_data, lb_targets = self.get_data()
            lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                                   self.transform, False, None, onehot)
            return lb_dset, None

        if self.name.upper() == 'STL10':
            lb_data, lb_targets, ulb_data = self.get_data()
            if include_lb_to_ulb:
                ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
            lb_data, lb_targets, _ = sample_labeled_data(self.args, lb_data, lb_targets, num_labels, self.num_classes)
            ulb_targets = None
        elif self.name.upper() == 'ATR':
            lb_data, lb_targets, ulb_data = self.get_atr_data(txt_data_dir, lb_files, ulb_files)
            ulb_targets = None
        else:
            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
                                                                        num_labels, self.num_classes,
                                                                        index, include_lb_to_ulb)
        # print("isTrain =", self.train, "Labeled data shape: ",lb_data.shape, "Unlabeled data shape: ", ulb_data.shape)
        # print("Classes: ",Counter(lb_targets))
        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(self.num_classes)]
        for c in lb_targets:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        # Supervised top line using all data as labeled data.
        if self.alg == 'fullysupervised' and (self.name.upper() == 'ATR'):
            #lb_data, lb_targets = self.get_data()
            lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                                   self.transform, False, None, onehot)
            return lb_dset, None
        # print(Counter(ulb_targets.tolist()))
        lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                               self.transform, False, None, onehot)

        ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes,
                                self.transform, True, strong_transform, onehot)
        # print(lb_data.shape)
        # print(ulb_data.shape)
        return lb_dset, ulb_dset
