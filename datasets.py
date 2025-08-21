import glob
import os
import cv2
import torch
import numpy
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data
import argparse
from torchsampler import ImbalancedDatasetSampler

# 使用与李珊等人相同的数据

def get_train_loader(args):
    train_domain = args.train_data.split(',')
    train_data = [ ]

    if "raf2" in train_domain:
        dataset0 = RAFTrainSet(args,  args.train_list0)
        '''
        使用样本不平衡采样器
        '''
        dataloader0 = DataLoader(
            dataset0,
            batch_size=args.batch_size,
            # sampler=ImbalancedDatasetSampler(dataset0),
            shuffle=args.shuffle,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True
        )
        train_data.append(dataloader0)
    return train_data

def get_test_loader(args): # testing data used in the training stage(different ck+ and jaffe)
    test_domain = args.test_data.split(',')
    test_data = []
    if 'jaf' in test_domain:
        # dataset6 = RAFTestSet(args,  args.test_list10)
        dataset6 = RAFTrainSet(args,  args.test_list10)
        data_loader6 = DataLoader(
            dataset6,
            batch_size=args.batch_size,
            # shuffle=False,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
        test_data.append(data_loader6)

    return test_data

def get_test_loader_pure_test(args):
    test_domain = args.test_data.split(',')
    test_data = []
    if 'jaf' in test_domain:
        dataset6 = RAFTestSet(args,  args.test_list6)
        data_loader6 = DataLoader(
            dataset6,
            batch_size=args.batch_size,
            shuffle=False,
            # shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader6)
    return test_data


class RAFTrainSet(data.Dataset):
    def __init__(self, args, data_list):
        self.images = glob.glob(data_list+'/*/*')
        # self.targets = [raf_labels.get(i.split('\\')[-2]) for i in self.images]
        # self.targets = [int(i.split('\\')[-2]) for i in self.images]
        self.targets = [int(i.split('/')[-2]) for i in self.images]
        self.args = args

        if self.args.aug == True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.Resize((256, 256)),
                # transforms.RandomCrop((224, 224)),
                transforms.Resize((124, 124)),
                transforms.RandomCrop((112, 112)),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((224, 224)),
                # transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image,(self.args.size,self.args.size))
        image = self.transform(image)
        target = self.targets[index]
        return image, target, index

    def get_labels(self):
        return self.targets

    def __len__(self):
        return len(self.targets)


class RAFTestSet(data.Dataset):
    def __init__(self, args, data_list):
        self.images = glob.glob(data_list+'/*/*')
        # self.targets = [jafe_labels.get(i.split('\\')[-2]) for i in self.images]
        # self.targets = [int(i.split('\\')[-2]) for i in self.images]
        self.targets = [int(i.split('/')[-2]) for i in self.images]
        self.args = args

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image,(self.args.size,self.args.size))
        image = self.transform(image)
        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.targets)

