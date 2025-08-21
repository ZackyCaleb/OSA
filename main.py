import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import args
from networks.DDAM import DDAMNet, pretrained_DCAMNF

from datasets import get_train_loader, get_test_loader, get_test_loader_pure_test
from log import Logger
from train import Trainer
import os
import torch
import datetime
import csv
import gc
# print("run date: " + str(datetime.datetime.now()))

import numpy as np
import random

# from Trans_resnet import Loss_aug_pro
os.environ["CUDA_VISIBLE_DEVICES"] = args.nGPU


# def seed_torch(seed=42):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True
# seed_torch(seed=3407)

def get_catalogue():
    model_creators = dict()
    model_creators['DDAMFN'] = DDAMNet

    return model_creators

def create_model(args):
    state = None

    model_creators = get_catalogue()        # 获取可用模型目录

    assert args.model in model_creators     # 确认参数中的模型在可用目录里

    model = model_creators['DDAMFN'](args)   # model结构
    if args.resume:     # 模型恢复
        save_path = os.path.join(args.save_path)  # 模型保存目录
        checkpoint = torch.load(save_path)

        model.load_state_dict(checkpoint['model'])      # 模型参数
        state = checkpoint['state']         # state参数

    cudnn.benchmark = True
    GPUs = args.nGPU.split(",")
    if len(GPUs) > 1:
        # 并行
        model = nn.DataParallel(model, device_ids=[i for i in range(len(GPUs))]).cuda()
        return model, state
    else:
        model = model.cuda()

        return model, state

def main():
    # Create Model, Criterion and State
    model, state = create_model(args)
    # isda_crit = ISDALoss(args.latent_dim, args.output_classes).cuda()
    # isda_crit = Loss_aug_pro(args.latent_dim, args.output_classes, 0.1).cuda()
    print("=> Model and criterion are ready")

    # Create Dataloader
    train_loader = get_train_loader(args)  # 获取训练数据, [dataloader1,dataloader2,dataloader3]
    val_loader = get_test_loader(args)  # 获取测试数据 [data_loader1,data_loader2,data_loader3,data_loader4]
    val_loader_pure_test = get_test_loader_pure_test(args)
    print("=> Dataloaders are ready")

    # Create Logger
    logger = Logger(args, state)            # 创建模型保存目录，记录state
    print("=> Logger is ready")

    # Create Trainer
    trainer = Trainer(args, model)
    print("=> Trainer is ready")
    print('=> super parameters: ' + str(vars(args)))

    if args.test_only:                      # 仅测试，前提是已经有训练好的模型了
       test_summary = trainer.test(0, val_loader)
       print("- Test:  Acc %6.3f " % (test_summary['acc']))
    else:   # 训练模式
        print(args.print)
        start_epoch = logger.state['epoch'] + 1     # 开始于上一次训练的下一个epoch
        print("=> Start training")

        domains = ['epoch', 'raf', 'aff', 'fer', 'ck+', 'mmi','jaf','oul','sfew']
        log_file = args.log_path+'.csv'
        with open(log_file, "a+", encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not args.target==None :
                # 先写入columns_name
                writer.writerow(['epoch', args.target])
                csvfile.close()
                best_epoches = [0, 0]
                best_accs = [0.0, 0.0]
            else:
                # 先写入columns_name
                writer.writerow(domains)
                csvfile.close()
                best_epoches = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                best_accs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        for epoch in range(start_epoch, args.n_epochs + 1):     # 在规定的训练epoch期间
            train_start = datetime.datetime.now()
            train_summary = trainer.train(epoch, train_loader, val_loader)

            test_start = datetime.datetime.now()
            test_summary = trainer.test(epoch, val_loader_pure_test)  # 测试一次 return only the acc
            test_end = datetime.datetime.now()
            domain_acc = test_summary['domain_acc']
            if not args.target==None :
                if domain_acc[1] > best_accs[1]:  # domain_acc[0] is the number of current epoch
                    best_accs[1] = domain_acc[1]
                    best_epoches[1] = epoch
                    best_accs[1] = domain_acc[1]
                    best_epoches[1] = epoch
                    best_model = model
                    best_epoch = epoch
                    best_test_summary = test_summary
                    best_train_summary = train_summary
                    logger.record(best_epoch, train_summary=best_train_summary, test_summary=best_test_summary,
                                  model=best_model)  # 记录当前最佳模型
            else:
                for index in range(8):
                    if domain_acc[index+1]>best_accs[index+1]:  # domain_Acc 第一个是epoch
                        best_accs[index+1] = domain_acc[index+1]
                        best_epoches[index+1] = epoch
                logger.record(epoch, train_summary=train_summary, test_summary=test_summary,
                              model=model)
            with open(log_file, "a+", encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(test_summary['domain_acc'])
                csvfile.close()
            print("training time of this epoch: " + str(test_start - train_start))
            print("testing time of this epoch: " + str(test_end - test_start))

        with open(log_file, "a+", encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(best_epoches)
            writer.writerow(best_accs)
            csvfile.close()

        logger.final_print()


if __name__ == '__main__':
    main()
