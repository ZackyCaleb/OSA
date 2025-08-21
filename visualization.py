'''
https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py
'''
import argparse
import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from networks.DDAM import DDAMNet

from torchvision import transforms
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import matplotlib.pyplot as plt
from typing import Union, List, Optional
import torch.nn.functional as F
from einops import rearrange
class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.cpu().detach().numpy()[0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    np_arr = (np_arr * 255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    if heatmap:
        np_arr = cv2.resize(np_arr, shape)
        np_arr = cv2.applyColorMap(np_arr, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    return np_arr / 255


def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
# 合并热力图和原题，并显示结果

# 获取热力图
def get_heatmap(activations):

    heatmap = torch.mean(activations, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap


def merge_heatmap_image(heatmap, image_path):
    img = cv2.imread(image_path)
    # img = cv2.resize(img, (224, 224))
    img = cv2.resize(img, (112, 112))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    grad_cam_img = heatmap * 0.5 + img
    grad_cam_img = grad_cam_img / grad_cam_img.max()
    # 可视化图像
    b,g,r = cv2.split(grad_cam_img)
    grad_cam_img = cv2.merge([r,g,b])

    # plt.figure(figsize=(8,8))
    # plt.imshow(grad_cam_img)
    # plt.axis('off')
    # plt.show()
    return grad_cam_img


def get_hot_map(path):

    target_image = cv2.imread(path)
    # target_image_name = opt.image_path.split('\\')[-1]
    preprocess_image = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((224, 224)),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # preprocess_image = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((112, 112)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])])
    target_input_tensor = preprocess_image(target_image).unsqueeze(0).cuda()

    feature_maps = {}
    def create_hook(name):
        def forward_hook(module, input, output):
            # 将输出保存到字典中，键为层的名称
            feature_maps[name] = output.detach().cpu()  # 如果在GPU上运行，则转移到CPU并分离梯度

        return forward_hook

    layers_to_hook = {

        'sour_encoder': model.sour_encoder
    }

    hooks = {}

    for name, layer in layers_to_hook.items():
        hook = layer.register_forward_hook(create_hook(name))
        hooks[name] = hook

    with torch.no_grad():  # 禁用梯度计算以节省内存和计算量
        output = model(target_input_tensor)


    # branch_1_out = feature_maps['dsa']
    # branch_2_out = feature_maps['cat_head1']
    branch_1_out = feature_maps['sour_encoder']
    # heat_map = get_heatmap(branch_1_out*branch_2_out)
    heat_map = get_heatmap(branch_1_out)
    pse_feat_map = merge_heatmap_image(heat_map, path)

    # print(output[1])
    for hook in hooks.values():
        hook.remove()

    return pse_feat_map, output[0]
    # return pse_feat_map, output[2]

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--raf_path', type=str, default=r'F:\project\DDAMFN-main\DDAMFN++\data\fer/', help='AfectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--model_path', default=r'..')
    return parser.parse_args()

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    import imageio

    '''
    hook获得深度特征
    '''
    # opt = get_args()
    # methods = {
    #     "gradcam": GradCAM,
    # }

    # fmap_block = list()
    # grad_block = list()
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''加载DDAMFN'''
    model = DDAMNet(num_class=7, num_head=args.num_head, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


    root = r'.\data\expw\test'
    save_tmp = r'.\ResNet_hot_maps\expw'

    MMI_all_image = glob.glob(root+'\*\*')
    for path in MMI_all_image:
        pse_feat_map, out = get_hot_map(path)
        pse_feat_map = (pse_feat_map * 255.0).astype('uint8')
        logit = torch.argmax(out).cpu().detach().numpy()
        label = int(path.split('\\')[-2])
        if logit == label:
            save_path_tmp = os.path.join(save_tmp, str(label))
            if not os.path.exists(save_path_tmp):
                os.mkdir(save_path_tmp)
            save_path = os.path.join(save_path_tmp, path.split('\\')[-1])
            # cv2.imwrite(save_path, pse_feat_map)
            imageio.imwrite(save_path, pse_feat_map)



