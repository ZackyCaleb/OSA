import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from recordermeter import RecorderMeter
from Trans_opts import args
from Trans_resnet import resnet50_with_adlayer, Net, MA_Net, CLUBSample
from Trans_datasets import get_train_loader, get_test_loader, get_test_loader_pure_test
from JDMAN_log import Logger
from Trans_train import Trainer
from torch.autograd import Variable

def get_catalogue():
    model_creators = dict()
    model_creators['resnet50_with_adlayer_all'] = resnet50_with_adlayer

    # model_creators['vgg16'] = vgg16_bn
    # model_creators['vgg16_local_global'] = VGG_local_global
    return model_creators


def create_model(args):

    model_creators = get_catalogue() # 获取可用模型目录

    assert args.model in model_creators # 确认参数中的模型在可用目录里

    model = model_creators['resnet50_with_adlayer_all'](args) # model结构

    # adv_model = AdversarialNetwork(3072, 512, args.n_epochs * 112)

    save_path = os.path.join(args.save_path, 'saco_50_acc_75.0.pth')  # 模型保存目录
    checkpoint = torch.load(save_path)

    model.load_state_dict(checkpoint['model'])  # 模型参数
    state = checkpoint['state']  # state参数

    val_loader_pure_test = get_test_loader_pure_test(args)
    test_loader = val_loader_pure_test[0]
    with torch.no_grad():
        # domain_acc = []
        # domain_acc.append(epoch)
        acc_avg = 0
        total = 0
        bingo = 0.
        model.eval()
        # model.cuda()
        for i, (input_tensor, target) in enumerate(test_loader):
            batch_size = target.size(0)
            # input_tensor = input_tensor.cuda()
            # target = target.cuda()
            input_var = Variable(input_tensor)
            target_var = Variable(target)
            mu_1, output = model(input_var)

            bingo += torch.sum(torch.argmax(output.data, dim=1) == target_var.data)

        acc_avg = bingo/len(test_loader.sampler)
        print("\n=>Acc %6.6f\n" % (acc_avg))

if __name__ == '__main__':

    create_model(args)