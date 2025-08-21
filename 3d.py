'''
https://github.com/Leethony/Additive-Margin-Softmax-Loss-Pytorch/tree/master
'''
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import torch
from datasets import get_train_loader, get_test_loader_pure_test
from opts import args
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import manifold
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from networks.DDAM import DDAMNet

def plot(embeds, labels, fig_path='./example.pdf'):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    # r = 1
    # pi = np.pi
    # cos = np.cos
    # sin = np.sin
    # phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0*pi:50j]
    # x = r*sin(phi)*cos(theta)
    # y = r*sin(phi)*sin(theta)
    # z = r*cos(phi)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)

    z = np.sqrt(x**2 + y**2)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)

def get_catalogue():
    model_creators = dict()
    # model_creators['resnet50_with_adlayer_all'] = resnet50_with_adlayer
    model_creators['DDAMFN'] = DDAMNet
    return model_creators

def creat_model(args):
    model_creators = get_catalogue()    # 获取可用模型目录

    assert args.model in model_creators     # 确认参数中的模型在可用目录里
    model = model_creators['resnet50_with_adlayer_all'](args)   # model结构

    save_path = r'..'
    checkpoint = torch.load(save_path)
    # checkpoint = torch.load(save_path, {'cuda:1': 'cuda:0'})
    # model.load_state_dict(checkpoint['model'])      # 模型参数
    model.load_state_dict(checkpoint['model'])  # 模型参数
    # state = checkpoint['state']  # state参数
    return model


def get_emb(args):
    val_loader_pure_test = get_test_loader_pure_test(args)
    # train_loader = get_train_loader(args)
    test_loader = val_loader_pure_test[0]
    # test_loader = train_loader[0]
    model = creat_model(args).cuda()
    target_full_emd = []
    target_full_label = []
    source_full_emd = []
    source_full_label = []
    bingo = 0.
    with torch.no_grad():
        model.eval()
        for i, (input_tensor, target) in enumerate(test_loader):
            input_tensor = input_tensor.cuda()
            input_var = Variable(input_tensor)
            target_var = Variable(target)
            # mu_1, output, var = model(input_var)
            # output, mu_1 = model(input_var)
            output, mu_1, ad_out_1 = model(input_var)
            target_full_emd.append(F.normalize(mu_1.detach().cpu()).numpy())
            # full_emd.append(mu_1.detach().cpu().numpy())
            target_full_label.append(target.numpy())
            # bingo += torch.sum(torch.argmax(output.data, dim=1) == target_var.data)
            bingo += torch.sum(torch.argmax(output.detach().cpu(), dim=1) == target_var.data)
        acc_avg = bingo/len(test_loader.sampler)
        print("\n=>KDEF Acc %6.6f\n" % (acc_avg))



    return np.vstack(target_full_emd), np.hstack(target_full_label)



def norm(reslut):
    reslut_min, reslut_max = reslut.min(0), reslut.max(0)
    reslut_norm = (reslut - reslut_min) / (reslut_max - reslut_min)  # 归一化
    return  reslut_norm

def get_label(label):
    get = []
    for i in range(len(label)):
        if label[i] == 0:
            get.append('Surprise')
        if label[i] == 1:
            get.append('Fear')
        if label[i] == 2:
            get.append('Disgust')
        if label[i] == 3:
            get.append('Happy')
        if label[i] == 4:
            get.append('Sad')
        if label[i] == 5:
            get.append('Anger')
        if label[i] == 6:
            get.append('Neutral')

    return get

if __name__ == '__main__':
    import scipy.io as scio
    import matplotlib.gridspec as gridspec  # 用网格来创建子图
    #
    # target_embeds, target_labels = get_emb(args)
    # # scio.savemat(r'affect-raf.mat', {'embed': target_embeds, 'label': target_labels})
    # scio.savemat(r'.\DDAMDN_MSceleb\sfew-raf.mat', {'embed':target_embeds, 'label':target_labels})


    '''
    加载数据
    '''

    target_data_tmp = scio.loadmat(r'.\DDAMDN_MSceleb\Oulu-oulu.mat')
    target_embeds, target_labels = target_data_tmp['embed'], target_data_tmp['label']

    source_data_tmp = scio.loadmat(r'.\DDAMDN_MSceleb\Oulu-raf.mat')
    source_embeds, source_labels = source_data_tmp['embed'], source_data_tmp['label']

    '''
    3-D可视化
    '''
    # tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    # target_X = tsne.fit_transform(source_embeds)
    # target_X = norm(target_X)
    # plot(target_X, source_labels, fig_path=r'F:\cross_domain_fer\papers\visualization/Raf_db_MMI.png')

    '''
    2D-可视化
    '''
    # palette = sns.color_palette("Paired", 12)
    palette = sns.color_palette("Paired", 7)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    target_y = np.squeeze(target_labels)
    target_y = get_label(target_y)
    target_X = tsne.fit_transform(target_embeds)
    target_X = norm(target_X)

    source_y = np.squeeze(source_labels)
    source_y = get_label(source_y)
    source_X = tsne.fit_transform(source_embeds)
    source_X = norm(source_X)

    pic = plt.figure(figsize=[5, 5])

    legend_font = {
        'family': 'Times New Roman',  # 字体
        'style': 'normal',
        'size': 6,  # 字号
        'weight': "normal",  # 是否加粗，不加粗
    }

    with sns.axes_style('ticks'):
        pic.add_subplot(1, 1, 1)
        # sns.scatterplot(x=target_X[:, 0], y=target_X[:, 1], hue=target_y, legend='full', palette=palette, markers='o')
        # sns.scatterplot(x=target_X[:, 0], y=target_X[:, 1], hue=target_y, markers='o')
        sns.scatterplot(x=target_X[:, 0], y=target_X[:, 1], hue=target_y, marker='^',  palette=palette,)
        # sns.scatterplot(x=target_X[:, 0], y=target_X[:, 1], hue=, marker='^',  palette=palette,)

        # sns.scatterplot(x=source_X[:, 0], y=source_X[:, 1], hue=source_y, legend='full', palette=palette, markers='s')
        # sns.scatterplot(x=source_X[:, 0], y=source_X[:, 1], hue=source_y, markers='s')
        sns.scatterplot(x=source_X[:, 0], y=source_X[:, 1], hue=source_y, marker='.', palette=palette,)
        plt.legend(loc="upper right", prop=legend_font, ncol=2, handletextpad=0.1, labelspacing=0.4, columnspacing=0.4)
        plt.rc('font', family='Times New Roman')
        plt.yticks(fontproperties='Times New Roman')  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman')
        bwith = 0.5  # 边框宽度设置为2
        TK = plt.gca()  # 获取边框
        TK.spines['bottom'].set_linewidth(bwith)
        TK.spines['left'].set_linewidth(bwith)
        TK.spines['top'].set_linewidth(bwith)
        TK.spines['right'].set_linewidth(bwith)
        TK.spines['bottom'].set_color('black')
        TK.spines['top'].set_color('black')
        TK.spines['left'].set_color('black')
        TK.spines['right'].set_color('black')

        img_path = r'.\DDAMDN_MSceleb\baseline_oulu.jpg'
        plt.savefig(img_path, dpi=800, bbox_inches='tight')
