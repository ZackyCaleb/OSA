import torch
import torch.optim as optim
from torch.autograd import Variable
import datetime
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from losses import mi_loss_mlp, L_je, JointMIEstimator

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.decay = 1
        self.model = model

        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.q_net = JointMIEstimator(feat_dim=512, num_classes=7).cuda()
        self.loss_fn = L_je(lambda_1=args.lambda_1, lambda_2=args.lambda_2)


        if args.opti == 'Adam':
            self.optimizer4nn = optim.Adam(
                list(model.parameters()) + list(self.q_net.parameters()),
                # list(model.parameters()) + list(self.q_net.parameters()),
                args.learn_rate,
                weight_decay=args.weight_decay,
                amsgrad=True
            )

        else:
            self.optimizer4nn = optim.SGD(
                list(model.parameters()) + list(self.q_net.parameters()),
                args.learn_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True
            )
        self.nGPU = args.nGPU
        self.learn_rate = args.learn_rate

    def train(self, epoch, train_loader, test_loader):
        # train_start = datetime.datetime.now()
        model = self.model
        torch.cuda.empty_cache()
        model.train()

        if self.args.opti == 'SGD':
        # if self.args.opti == 'Adam':
            self.learning_rate(epoch)
            # with target data
            # the second one is the target domain
            len_dataloader1 = len(train_loader[0])
            len_dataloader2 = len(test_loader[0])
            data_1_iter = iter(train_loader[0])
            data_2_iter = iter(test_loader[0])
            threshold = 0.5
            i = 1
            j = 1
            while i < len_dataloader1:
                # Training model using 1st dataset
                # data_1 = data_1_iter.next()
                data_1 = next(data_1_iter)
                data_1_img, data_1_label = data_1[0].cuda(), data_1[1].cuda()
                '''
                原来的 
                '''
                # mu_1, output_1, ad_out_1 = model(data_1_img, epoch, True)
                # output_1, mu_1 = model(data_1_img)
                output_1, mu_1, ad_out_1 = model(data_1_img)

                if j < len_dataloader2:
                    # data_2 = data_2_iter.next()
                    data_2 = next(data_2_iter)
                    j += 1
                else:
                    data_2_iter = iter(test_loader[0])
                    j = 1
                    # data_2 = data_2_iter.next()
                    data_2 = next(data_2_iter)

                '''
                原来的 
                '''
                data_2_img, _ = data_2[0].cuda(), data_2[1].cuda()
                # mu_2, output_2, ad_out_2 = model(data_2_img, epoch, True)
                # output_2, mu_2 = model(data_2_img)
                output_2, mu_2, ad_out_2 = model(data_2_img)

                train_acc_1 = self.accuracy(output_1.data, data_1_label.data, (1,))[0]
                loss_1 = self.criterion_cls(output_1, data_1_label)
                # en_loss = entropy_loss(output_2)
                # en_loss = information_maximization_loss(output_2)

                labels_target_fake = torch.max(torch.nn.Softmax(dim=1)(output_2), 1)[1]
                ad_out = torch.cat([ad_out_1, ad_out_2], dim=0)
                # logit = torch.cat([output_1, output_2], dim=0)
                mi_loss = mi_loss_mlp(mu_1, mu_2, data_1_label, labels_target_fake, self.q_net)

                '''Joint entropy loss'''
                if epoch >= 30:
                    je_loss = self.loss_fn.estimated(output_2)
                else:
                    je_loss = 0.
                # # je_loss = loss_fn.estimated(output_2)
                # # je_loss = logit_kl(mu_1, mu_2, data_1_label, labels_target_fake)
                '''Cross-entropy loss'''
                exp_loss = self.criterion_cls(output_1, data_1_label)
                loss = exp_loss + mi_loss + je_loss

                self.optimizer4nn.zero_grad()
                loss.backward()
                self.optimizer4nn.step()

                # if self.args.MI:
                #     a1, b1 = mu_1.shape
                #     a2, b2 = mu_2.shape
                #     features = torch.cat((mu_1, mu_2), dim=0)
                #     labels = torch.cat((data_1_label, labels_target_fake), -1).cuda()
                #     # outputs = torch.cat((output_1, output_2), dim=0)
                #     # softmax_out_2 = torch.nn.Softmax(dim=1)(output_2)
                #     # dc_target = torch.from_numpy(np.array([[1]] * a1 + [[0]] * a2)).float().cuda()
                #     dc_target = torch.from_numpy(np.array([[1]] * a1 + [[0]] * a2)).cuda()
                #
                #     if self.args.model == 'resnet50_with_adlayer_all':
                #         adv_loss = self.club_net(features, dc_target.squeeze(dim=-1))
                #         loss = loss_1 + adv_loss
                #         # loss = loss_1 + mi_lb
                #         # loss = loss_1
                #         self.optimizer4nn.zero_grad()
                #         loss.backward()
                #         # self.optimizer4nn.step()
                #         self.optimizer4nn.first_step(zero_grad=True)
                #         output_1, mu_1, ad_out_1 = model(data_1_img)
                #         output_2, mu_2, ad_out_2 = model(data_2_img)
                #         loss_1 = self.criterion_class(output_1, data_1_label)
                #         ad_out = torch.cat([ad_out_1, ad_out_2], dim=0)
                #         # dc_target = torch.from_numpy(np.array([[1]] * a1 + [[0]] * a2)).float().cuda()
                #         dc_target = torch.from_numpy(np.array([[1]] * a1 + [[0]] * a2)).cuda()
                #         adv_loss = self.bce(ad_out, dc_target)
                #
                #         adv_loss = self.club_net(features, dc_target.squeeze(dim=-1))
                #
                #         loss = loss_1 + adv_loss
                #         # loss = loss_1
                #         self.optimizer4nn.zero_grad()
                #         loss.backward()
                #         self.optimizer4nn.second_step(zero_grad=True)

                if (i - 1) % 30 == 0:
                    print(
                        'Ep:[{}/{}], Ba:[{}/{}], ls1:{:.4f}, lmi:{:.4}, lje:{:.4}, ac1:{:.4f}, lr:{:.4f}'.format(
                            epoch, self.args.n_epochs, i, len_dataloader1, exp_loss.item(),
                            # adv_loss.item(), mi_lb.item(), train_acc_1.item(), self.decay * self.learn_rate))
                            mi_loss.item(), je_loss.item(), train_acc_1.item(), self.decay * self.learn_rate))

                i += 1
            summary = dict()
            summary['acc'] = 0.0  # no use at all
        return summary

    def test(self, epoch=0, test_loader=None):
        with torch.no_grad():
            domain_acc = []
            domain_acc.append(epoch)
            if self.args.target==None: # testing on all datasets
                domains = ['raf', 'aff', 'fer', 'ck+', 'mmi', 'jaf', 'oul', 'sfew']

                model = self.model
                model.eval()
                for dom in range(len(test_loader)):
                    acc_avg = 0
                    total = 0
                    predicted_list = []
                    target_list = []
                    for i, (input_tensor, target) in enumerate(test_loader[dom]):

                        if not self.nGPU==None:
                            input_tensor = input_tensor.cuda()
                            target = target.cuda()

                        batch_size = target.size(0)
                        input_var = Variable(input_tensor)
                        # mu_1, output, var = model(input_var)
                        # output, mu_1 = model(input_var, epoch)
                        output, mu_1, var = model(input_var)

                        acc = self.accuracy(output.data, target, (1,))[0]
                        acc_avg += acc * batch_size

                        _, predicted = torch.max(output.data, 1)
                        predicted_list.extend(list(predicted.data))
                        target_list.extend(list(target.data))
                        total += batch_size

                    acc_avg /= total
                    print("ACC on :  " + domains[dom]+'\n')
                    # print(self.get_confuse_matrix(predicted_list,target_list))
                    print("\n=> Test[%d]  Acc %6.3f\n" % (epoch, acc_avg))
                    domain_acc.append(acc_avg.item())

                torch.cuda.empty_cache()

                summary = dict()

                summary['acc'] = 0.0                # nouse
                summary['domain_acc'] = domain_acc
            else:
                acc_avg = 0
                total = 0
                predicted_list = []
                target_list = []
                model = self.model
                model.eval()
                for i, (input_tensor, target) in enumerate(test_loader[0]):

                    if not self.nGPU==None:
                        input_tensor = input_tensor.cuda()
                        target = target.cuda()

                    batch_size = target.size(0)
                    input_var = Variable(input_tensor)

                    output, _, var = model(input_var)

                    acc = self.accuracy(output.data, target,(1,))[0]
                    acc_avg += acc * batch_size

                    _, predicted = torch.max(output.data, 1)
                    predicted_list.extend(list(predicted.data))
                    target_list.extend(list(target.data))

                    total += batch_size

                acc_avg /= total
                print("ACC on :  " + self.args.target+'\n')
                # print(self.get_confuse_matrix(predicted_list,target_list))
                print("\n=> Test[%d]  Acc %6.3f\n" % (epoch, acc_avg))
                domain_acc.append(acc_avg.item())

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = 0.0  # nouse
        summary['domain_acc']= domain_acc
        return summary

    def get_test_features(self, epoch, test_loader):

        n_batches = len(test_loader[0])

        acc_avg = 0.0
        total = 0

        model = self.model
        model.eval()
        predicted_list = []
        target_list = []
        feature_list = []
        for i, (input_tensor, target) in enumerate(test_loader[0]):

            if not self.nGPU==None :
                input_tensor = input_tensor.cuda()
                target = target.cuda()

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            # target_var = Variable(target)

            if self.args.criterion == 'DLP_LOSS':
                output, exp_f, feature = model(input_var)
            else:
                output, exp_f, feature = model(input_var)
            feature_list.extend(list(feature.cpu().detach().numpy()))

            acc = self.accuracy(output.data, target, (1,))[0]
            acc_avg += acc * batch_size

            _, predicted = torch.max(output.data, 1)
            predicted_list.extend(list(predicted.data))
            target_list.extend(list(target.cpu().detach().numpy().tolist()))

            total += batch_size
            if i % 100 == 0:
                print("| Test[%d] [%d/%d]   Acc %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    acc_avg / total))

        acc_avg /= total
        print("\n=> Test[%d]  Acc %6.3f\n" % (
            epoch,
            acc_avg))

        summary = dict()

        summary['acc'] = acc_avg  # testset
        return summary, feature_list, target_list

    def get_confuse_matrix(self, predicted, target):
        np.set_printoptions(suppress=True,precision=4)
        num = int(self.args.output_classes)
        con_mat = np.zeros((num, num), np.float)
        for index in range(len(target)):
            con_mat[target[index]][predicted[index]] += 1
        for i in range(num):
            a = np.sum(con_mat, axis=1)  # sum of every row according to ECAN
            for j in range(num):
                con_mat[i][j] /= a[i]
        return con_mat

    def accuracy(self,output,target,topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correcct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correcct_k = correcct[:k].view(-1).float().sum(0,keepdim=True)
            res.append(correcct_k.mul_(100.0 / batch_size)[0])
        return res  # the largest

    def learning_rate(self, epoch):
        self.decay = 0.1 **((epoch - 1) // self.args.decay)
        learn_rate = self.learn_rate * self.decay
        if learn_rate < 1e-7:
            learn_rate = 1e-7
        for param_group in self.optimizer4nn.param_groups:
        # for param_group in self.optimizer4center.param_groups:
            param_group['lr'] = learn_rate

