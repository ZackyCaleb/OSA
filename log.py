import glob
import os
import torch
import numpy as np


class Logger:
    def __init__(self, args, state):
        self.args = args
        if not state:
            self.state = dict()
            self.state['epoch'] = 0  # 记录最佳epoch和acc
            self.state['best_acc'] = 0
        else:
            self.state = state

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)


        self.save_path = args.save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)  # 创建保存目录如 :/.../resnet18

        if args.train_record and not args.test_only:
            self.train_record = [] # 如果不是仅测试模型，并且记录训练，这里记录什么东西？？？？？？？？
        else:
            self.train_record = None

    def record(self, epoch, train_summary=None, test_summary=None, model=None):
        # 训练时记录训练和测试summery，最佳模型的epoch和对应的最佳acc，记录
        assert train_summary != None or test_summary != None, 'Need at least one summary'

        if torch.typename(model).find('DataParallel') != -1:
            model = model.module # 并行情况下model.module下才是模型的结构

        self.state['epoch'] = epoch # 当前进行的epoch

        if train_summary:
            # latest = os.path.join(self.save_path, 'latest.pth') # record which is the latest epoch.
            # torch.save({'latest': epoch}, latest)
            weight_files = glob.glob(self.save_path+'/*.pth')
            f = weight_files[0]
            # for f in weight_files:
            f = f[:-4]
            acc = float(f.split('_')[-1])
            if acc<=test_summary['domain_acc'][1]:
                os.remove(f+'.pth')
                model_file = os.path.join(self.save_path, 'OSA_{}_acc_{}.pth'.format(epoch, test_summary['domain_acc'][1]))

                checkpoint = dict()
                checkpoint['state'] = self.state  # 更新state
                checkpoint['model'] = model.state_dict()  # 更新模型

                torch.save(checkpoint, model_file)  # 保存模型参数和state

                keys = train_summary.keys()
                keys = sorted(keys)
                train_rec = [train_summary[key] for key in keys]  # 按键排序 记录train summery 的值

        if test_summary:
            top = test_summary['acc']
            state_top = self.state['best_acc']  # 以前保存的最好结果

            if top >= state_top:  # 如果本次测试最大，则更新best_acc,并记录对应的best_epoch
                self.state['best_acc'] = test_summary['acc']
                self.state['best_epoch'] = epoch

                # best = os.path.join(self.save_path, 'best.pth')
                # torch.save({'best': epoch}, best)  # 这里只记录最佳的epoch

            keys = test_summary.keys()
            keys = sorted(keys)
            test_rec = [test_summary[key] for key in keys] # test的summery值

        if self.train_record is not None:
            self.train_record.append(train_rec + test_rec)  # 记录训练和测试的summery的值
            record = os.path.join(self.save_path, 'train.record')
            np.save(record, self.train_record)

    def final_print(self):
        print("- Best:  Acc %6.3f at %d" % (
            self.state['best_acc'],
            self.state['best_epoch']))
