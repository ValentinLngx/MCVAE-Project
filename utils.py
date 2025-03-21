import argparse
import os
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def repeat_data(x, n_samples):
    if x.dim() == 4:
        return x.repeat(n_samples, 1, 1, 1)
    else:
        return x.repeat(n_samples, 1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MyDataset(Dataset):
    def __init__(self, data, binarize=False, reshape=False):
        super(MyDataset, self).__init__()
        if isinstance(data, list):
            # Always assume the first element is the data and the second element (if available) is labels
            self.data = data[0]
            self.labels = data[1] if len(data) > 1 else None
        else:
            self.data = data
            self.labels = None
        self.binarize = binarize
        self.reshape = reshape

        if isinstance(self.data, np.ndarray):
            self.shape_size = self.data.shape[-2]
        elif isinstance(self.data[0], str):
            self.shape_size = 64
        #else:
        #    self.shape_size = self.data[0][0].size()[-1]
        else:
            if isinstance(self.data[0], torch.Tensor):
                if self.data[0].dim() == 2:
                    self.shape_size = self.data[0].shape[0]
                elif self.data[0].dim() >= 3:
                    self.shape_size = self.data[0].shape[-2]
                else:
                    self.shape_size = self.data[0].size()[-1]
            else:
                self.shape_size = self.data[0][0].size()[-1]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if isinstance(self.data[item], tuple):
            sample, _ = self.data[item]
        else:
            sample = self.data[item]
        
        if self.reshape:
            sample = Image.open(sample).convert("RGB")
            sample = transforms.Compose([
                transforms.Resize((self.shape_size, self.shape_size)),
                transforms.ToTensor(),
            ])(sample)
        else:
            if not isinstance(sample, torch.Tensor):
                sample = transforms.ToTensor()(sample)
            else:
                if sample.dim() == 2:
                    sample = sample.unsqueeze(0)
        if self.binarize:
            sample = torch.distributions.Bernoulli(probs=sample).sample()
        label = -1. if self.labels is None else self.labels[item]
        return sample, label


def make_dataloaders(dataset, batch_size, val_batch_size, binarize=False, **kwargs):
    if dataset == 'mnist':
        data = datasets.MNIST('./data', train=True, download=True)
        train_data = data.train_data.numpy()
        train_labels = data.train_labels.numpy()
        train_dataset = MyDataset([train_data, train_labels], binarize=binarize)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        val_data = data.test_data.numpy()
        val_labels = data.test_labels.numpy()
        val_dataset = MyDataset([val_data, val_labels], binarize=binarize)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,persistent_workers=True, **kwargs)

    elif dataset == 'fashionmnist':
        train_data = datasets.FashionMNIST('./data', train=True, download=True).train_data.to(torch.float32)
        train_data /= train_data.max()
        train_dataset = MyDataset(train_data, binarize=binarize)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        val_data = datasets.FashionMNIST('./data', train=False).test_data.to(torch.float32)
        val_data /= val_data.max()
        val_dataset = MyDataset(val_data, binarize=binarize)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,persistent_workers=True, **kwargs)

    elif dataset == 'cifar':
        train_data = datasets.CIFAR10('./data', train=True, download=True).data
        train_dataset = MyDataset(train_data, binarize=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        val_data = datasets.CIFAR10('./data', train=False, download=True).data
        val_dataset = MyDataset(val_data, binarize=False)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, persistent_workers=True,**kwargs)

    elif dataset == 'omniglot':
        train_data = datasets.Omniglot('./data', background=True, download=True)
        train_dataset = MyDataset(train_data, binarize=binarize)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True,**kwargs)

        val_data = datasets.Omniglot('./data', background=False, download=True)
        val_dataset = MyDataset(val_data, binarize=binarize)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, persistent_workers=True,**kwargs)

    elif dataset == 'celeba':
        path = './data/celeba/img_align_celeba_png.7z/img_align_celeba_png'
        train_data = [os.path.join(path, s) for s in os.listdir(path) if s.endswith('.png')]
        shuffle(train_data)
        n_objects = len(train_data)
        val_data = train_data[int(n_objects * 0.8):]
        train_data = train_data[:int(n_objects * 0.8)]
        train_loader = DataLoader(MyDataset(train_data, binarize=False, reshape=True), batch_size=batch_size,
                                  shuffle=True, **kwargs)
        val_loader = DataLoader(MyDataset(val_data, binarize=False, reshape=True), batch_size=val_batch_size,
                                shuffle=False, **kwargs)
    else:
        raise ValueError

    return train_loader, val_loader


def get_activations():
    return {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "logsoftmax": lambda: torch.nn.LogSoftmax(dim=-1),
        "logsigmoid": torch.nn.LogSigmoid,
        "softplus": torch.nn.Softplus,
        "gelu": nn.GELU
    }



def binary_crossentropy_logits_stable(logits, targets):
    abs_logits = torch.abs(logits)
    bce = torch.clamp(logits, min=0) - logits * targets + torch.log(1.0 + torch.exp(-abs_logits))
    return bce