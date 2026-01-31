import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from recordermeter import RecorderMeter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pre_process import ResizeImage

def image_classification_test(loader, model, test_10crop=True):
    model.eval()
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                # data = [iter_test[j].next() for j in range(10)]
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            # iter_test = iter(loader["test"])
            iter_test = iter(loader)
            # for i in range(len(loader['test'])):
            for i in range(len(loader)):
                # data = iter_test.next()
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict.cuda()).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

if __name__ == '__main__':
    base_network = torch.load(r'F:\project\CDAN-master\pytorch\snapshot\mmi\0.552best_model.pth.tar')
    dsets = ImageFolder(root=r'F:\project\DIFC\data\raf\test', transform= transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    dset_loaders = DataLoader(dsets, batch_size=4, shuffle=False, num_workers=4)
    temp_acc = image_classification_test(dset_loaders, base_network, test_10crop=False)
    print(temp_acc)