'''
https://blog.csdn.net/qq_40243750/article/details/124255865
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import os
from recordermeter import RecorderMeter
from Trans_resnet import resnet50_with_adlayer, jdman_resnet50_with_adlayer
from JDMAN_datasets import get_test_loader_pure_test
from torch.autograd import Variable
from Trans_opts import args
from matplotlib import rcParams
os.environ["CUDA_VISIBLE_DEVICES"] = args.nGPU
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
    # 设置西文字体为新罗马字体


    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "axes.unicode_minus": False  # 解决负号无法显示的问题
    }
    rcParams.update(config)

    plt.imshow(cm*100, cmap='Blues')
    # plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    # plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            # value = float(format('%.4f' % cm[j, i]))
            value = round(cm[j, i]*100, 2)
            # value = cm[j, i]
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

def get_catalogue():
    model_creators = dict()
    # model_creators['resnet50_with_adlayer_all'] = resnet50_with_adlayer
    model_creators['resnet50_with_adlayer_all'] = jdman_resnet50_with_adlayer
    # model_creators['vgg16'] = vgg16_bn
    # model_creators['vgg16_local_global'] = VGG_local_global
    return model_creators

def create_model(args):

    model_creators = get_catalogue()        # 获取可用模型目录

    assert args.model in model_creators     # 确认参数中的模型在可用目录里

    model = model_creators['resnet50_with_adlayer_all'](args)           # model结构

    # adv_model = AdversarialNetwork(3072, 512, args.n_epochs * 112)

    # save_path = os.path.join(args.save_path, 'ESPnet_42_acc_78.4000015258789.pth')  # 模型保存目录
    # save_path = os.path.join(args.save_path, 'ESPnet_18_acc_71.93877410888672.pth')  # 模型保存目录
    # save_path = os.path.join(args.save_path, 'ESPnet_40_acc_73.9795913696289.pth')  # 模型保存目录
    # save_path = os.path.join(args.save_path, 'ESPnet_59_acc_67.5.pth')  # 模型保存目录
    save_path = os.path.join(args.save_path, 'JMME', 'JMME_42_acc_52.663978576660156.pth')  # 模型保存目录
    checkpoint = torch.load(save_path)

    model.load_state_dict(checkpoint['model'])  # 模型参数
    state = checkpoint['state']  # state参数

    val_loader_pure_test = get_test_loader_pure_test(args)
    test_loader = val_loader_pure_test[0]
    y_gt = []
    y_pred = []
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
            # _, mu_1, output, var = model(input_var)
            # output, mu_1 = model(input_var)
            output, mu, ad_out_1 = model(input_var)
            bingo += torch.sum(torch.argmax(output.data, dim=1) == target_var.data)
            predict_np = np.argmax(output.cpu().detach().numpy(), axis=-1)  # array([0,5,1,6,3,...],dtype=int64)
            labels_np = target.numpy()  # array([0,5,0,6,2,...],dtype=int64)
            y_pred.append(predict_np)
            y_gt.append(labels_np)

        acc_avg = bingo/len(test_loader.sampler)
        print("\n=>Acc %6.6f\n" % (acc_avg))
    return np.hstack(y_pred), np.hstack(y_gt)

if __name__ == '__main__':

    y_pred, y_gt = create_model(args)
    draw_confusion_matrix(y_gt, y_pred, label_name=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"],
                          # title="KDEF w/o pop", pdf_save_path=r'F:\cross_domain_fer\papers/MMI_ESPnet.jpg', dpi=800)
                          # title="KDEF w/o pop", pdf_save_path=r'F:\cross_domain_fer\papers/visualization/KDEF_ESPnet.jpg', dpi=800)
                          # title="KDEF w/o pop", pdf_save_path=r'F:\cross_domain_fer\papers/visualization/JAFFE_ESPnet.jpg', dpi=800)
                          # pdf_save_path=r'F:\cross_domain_fer\papers\First\Trans\TLA\figs\fusion_matrix/Raf-expw.jpg', dpi=800)
                          pdf_save_path=r'F:\cross_domain_fer\papers\First\Trans\ESPnet\JMME_figs\fusion_matrix/fer2exp.jpg', dpi=800)