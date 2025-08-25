from torch import nn
import torch
from networks.MixedFeatureNet import MixedFeatureNet
from torch.nn import Module
import os
import math
import torch.nn.functional as F
# from copy import deepcopy
# from collections import OrderedDict
# from functools import partial
# from .deform_conv_v2 import DeformConv2d
# from einops import rearrange
from collections import OrderedDict

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DCA(nn.Module):
    def __init__(self, in_channels, n_directions=2, reduction=32):
        super().__init__()
        self.n_directions = n_directions
        self.in_channels = in_channels

        # 方向参数生成器
        self.dir_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, 512, 1, 1)
            nn.Conv2d(in_channels, in_channels // reduction, 1),    # (B, 512/32, 1, 1)
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, n_directions * 2, 1)  # 输出角度和缩放因子   # (B, 3*2, 1, 1)
        )

        # 特征变换层
        mid_channels = max(8, in_channels // reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        # 注意力生成层（每个方向独立）
        self.att_convs = nn.ModuleList([
            nn.Conv2d(mid_channels, in_channels, 1) for _ in range(n_directions)
        ])

        # 方向融合参数
        self.fusion = nn.Parameter(torch.ones(n_directions) / n_directions, requires_grad=True) # 学习对应方向的参数

    def _dynamic_pooling(self, x, theta, scale):
        N, C, H, W = x.size()
        device = x.device

        # 生成方向网格
        directions = []
        for d in range(self.n_directions):
            # 计算方向向量
            angle = theta[:, d] * math.pi  # 角度范围[0, π]
            cos_theta = torch.cos(angle).view(N, 1, 1, 1)  # [N,1,1,1]=(B,1,1,1)
            sin_theta = torch.sin(angle).view(N, 1, 1, 1)   # (B,1,1,1)

            # 生成可变形网格
            grid = self._create_deform_grid(H, W, cos_theta, sin_theta, scale[:, d]).to(device)     # 旋转后的坐标系=(128,7,7,2)

            # 应用可变形池化
            pooled = F.grid_sample(x, grid, align_corners=False)    # 使用双线性插值计算X中指定坐标的值
            directions.append(pooled)

        return torch.stack(directions, dim=1)  # [N,D,C,H,W]

    def _create_deform_grid(self, H, W, cos_theta, sin_theta, scale):
        N = cos_theta.size(0)

        # 基础坐标网格
        base_grid = torch.meshgrid(
            torch.linspace(-1, 1, H),   # 代表纵坐标
            torch.linspace(-1, 1, W),   # 代表横坐标
        indexing='ij')      # (7, 7), (7, 7)
        grid = torch.stack(base_grid, dim=-1).unsqueeze(0).repeat(N, 1, 1, 1).cuda()  # [N,H,W,2]=(B, 7, 7, 2):B个特征每个像素的空间坐标
        # grid = torch.stack(base_grid, dim=-1).unsqueeze(0).repeat(N, 1, 1, 1).to('cuda:1')  # [N,H,W,2]=(B, 7, 7, 2):B个特征每个像素的空间坐标

        # 应用旋转和缩放
        rot_matrix = torch.cat([
            cos_theta, -sin_theta,
            sin_theta, cos_theta
        ], dim=-1).view(N, 2, 2).cuda()  # 旋转矩阵 # (B, 1, 1, 4)-->(B, 2, 2)
        # ], dim=-1).view(N, 2, 2).to('cuda:1')  # 旋转矩阵 # (B, 1, 1, 4)-->(B, 2, 2)

        scaled_grid = torch.einsum('nij,nhwj->nhwi', rot_matrix, grid) * scale.view(N, 1, 1, 1).cuda()  # 得到旋转后的坐标系=(B, 7, 7, 2)
        # scaled_grid = torch.einsum('nij,nhwj->nhwi', rot_matrix, grid) * scale.view(N, 1, 1, 1).to('cuda:1')  # 得到旋转后的坐标系=(B, 7, 7, 2)
        return scaled_grid

    def forward(self, x):
        identity = x    # (B, 512, 7, 7)
        N, C, H, W = x.size()

        dir_params = self.dir_generator(x)  # [N, 2*D, 1, 1]=(B, 6, 1, 1)
        dir_params = dir_params.view(N, self.n_directions, 2)   # (B, 3, 2)
        theta = torch.sigmoid(dir_params[:, :, 0])  # 角度参数 [0,1] -> [0, π]=(B, 3)
        scale = 0.5 + torch.sigmoid(dir_params[:, :, 1])  # 缩放因子 [0.5, 1.5]=(B, 3)

        pooled = self._dynamic_pooling(x, theta, scale)  # [N,D,C,H,W]=(B,3,512,7,7)
        pooled = pooled.view(N * self.n_directions, C, H, W)    # (3B, 512, 7, 7)

        transformed = self.conv(pooled)  # [N*D, mid, H, W]     # (3B, 16, 7, 7)

        # 生成各方向注意力图
        atts = []
        for d in range(self.n_directions):
            att = self.att_convs[d](transformed[d * N:(d + 1) * N])  # [N,C,H,W]    # 对每个方向卷积生成独立的注意力图
            atts.append(att.sigmoid())

        fused_att = torch.zeros_like(x)
        for d in range(self.n_directions):
            weight = self.fusion[d].sigmoid()  # 方向权重
            fused_att += weight * atts[d]

        return identity * fused_att

# class DDAMNet(nn.Module):
#     def __init__(self, num_class=7, num_head=2, pretrained=True):
#         super(DDAMNet, self).__init__()
#
#         net = MixedFeatureNet.MixedFeatureNet()
#
#         if pretrained:
#             # net = torch.load(os.path.join('../pretrained/', "MFN_msceleb.pth"))
#             net = torch.load(r'./pretrained/MFN_msceleb.pth')
#
#         self.features = nn.Sequential(*list(net.children())[:-4])
#         self.num_head = num_head
#         for i in range(int(num_head)):
#             setattr(self,"cat_head%d" %(i), CoordAttHead())
#         # self.dsa = DCA(in_channels=512, n_directions=num_head)
#         self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
#         self.flatten = Flatten()
#         self.fc = nn.Linear(512, num_class)
#         self.bn = nn.BatchNorm1d(num_class)
#         self.cla_do = nn.Sequential(
#             nn.Linear(519, 1),
#             nn.Sigmoid())
#     def forward(self, x):
#         x = self.features(x)
#         heads = []
#
#         for i in range(self.num_head):
#             heads.append(getattr(self,"cat_head%d" %i)(x))
#         head_out = heads
#
#         y = heads[0]
#
#         for i in range(1, self.num_head):
#             y = torch.max(y, heads[i])
#
#         y = x*y
#         '''原来的'''
#         # y = self.dsa(x)
#         y = self.Linear(y)
#         y = self.flatten(y)
#         out = self.fc(y)
#         # return out, x, head_out
#         return out, x, None
#         '''用于Joint mutual information的'''
#         # y = self.Linear(y)
#         # f = self.flatten(y)
#         # logit_em = self.fc(f)
#         # # return out, x, head_out
#         # # return out, x, None
#         # d_f = torch.cat((f, logit_em), 1)
#         # logit_do = self.cla_do(d_f)
#         # return logit_em, f, logit_do

class Attention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.key = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, dim)
        )
        self.value = nn.Linear(dim, dim)
        self.attn = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        # self.attn = DeformConv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape    # (B, C, H, W)
        shortcut = x.clone()
        prem = x.permute(0, 2, 3, 1)
        value = self.value(prem).reshape(b, -1, c)
        key = self.key(prem) * prem
        key = key.reshape(b, -1, c)
        attn = self.attn(x).permute(0, 2, 3, 1)
        attn = self.norm(self.act(attn)).reshape(b, -1, c)
        attn = self.softmax(attn + key) * value
        attn = attn.reshape(b, h, w, c)
        attn = attn.permute(0, 3, 1, 2)
        return attn + shortcut

class PC_Attention(nn.Module):
    """Position-Content Attention Module with dynamic and static feature fusion.

    Args:
        dim (int): Input feature dimension.
        kernel_size (int): Convolution kernel size for static attention. Default: 7.
        patch_size (int): Window size for residual computation. Default: 7.
    """

    def __init__(self, dim, kernel_size=7, patch_size=7, kernel_sizes=[1,3,5]):
        super().__init__()
        # Dynamic attention components
        self.key_proj = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, dim))
        self.value_proj = nn.Linear(dim, dim)

        # Static attention components
        self.static_attn = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=1)

        # Residual path components
        self.patch_size = patch_size
        # self.residual = nn.Sequential(
        #     nn.Conv1d(dim, dim, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv1d(dim, dim * 9, kernel_size=1))  # 9=3x3 conv kernels
        self.final_norm = nn.LayerNorm(dim)
        # self.channel_attn = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim // 16, 1),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 16, dim, 1),
        #     nn.Sigmoid())
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 16, k, padding=k // 2, bias=False),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        self.fusion = nn.Conv2d(len(kernel_sizes) * (dim // 16), dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            Tensor: Output feature map of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Input dimensions must be divisible by patch_size"
        # shortcut = x.clone()
        shortcut = x

        # ========== Spatial Context Attention Computation ==========
        # Prepare feature maps
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Dynamic attention path
        value = self.value_proj(x_perm).reshape(B, -1, C)  # (B, H*W, C)
        key = self.key_proj(x_perm) * x_perm  # Dynamic gating
        key = key.reshape(B, -1, C)  # (B, H*W, C)

        # Static attention path
        static_attn = self.static_attn(x).permute(0, 2, 3, 1)  # (B, H, W, C)
        static_attn = self.norm(self.act(static_attn)).reshape(B, -1, C)

        # Attention fusion
        spatial = self.softmax(static_attn + key) * value  # (B, H*W, C)
        spatial = spatial.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # ========== Channel Context Attention Computation ==========
        # feat_a = x.flatten(2)
        # channel_attn = torch.matmul(feat_a, feat_a.transpose(1,2))
        # channel_attn = channel_attn.softmax(dim=-1)
        # channel = (channel_attn @ feat_a).reshape(B, C, H, W)
        # # ========== Residual Path ==========
        # # # Window partition
        # # num_windows_h = H // self.patch_size
        # # num_windows_w = W // self.patch_size
        # # window = rearrange(
        # #     shortcut,
        # #     'b c (h_win p1) (w_win p2) -> (b h_win w_win) p1 p2 c',
        # #     p1=self.patch_size,
        # #     p2=self.patch_size,
        # #     h_win=num_windows_h,
        # #     w_win=num_windows_w
        # # )  # (B*num_windows, patch_size, patch_size, C)
        # #
        # # # Residual weight generation
        # # window_flat = window.reshape(-1, self.patch_size * self.patch_size, C)
        # # B_win, N_win, C_win = window_flat.shape
        # #
        # # # Generate 3x3 conv weights from window statistics
        # # res_weight = self.residual(
        # #     window_flat.mean(dim=1).unsqueeze(-1)  # (B_win, C, 1)
        # # ).reshape(B_win * C_win, 1, 3, 3)  # (B_win*C, 1, 3, 3)
        # #
        # # # Apply depth-wise convolution
        # # shortcut = rearrange(
        # #     window_flat,
        # #     '(b h w) (p1 p2) c -> b (h w c) p1 p2',
        # #     p1=self.patch_size,
        # #     p2=self.patch_size,
        # #     h=num_windows_h,
        # #     w=num_windows_w,
        # #     b=B
        # # )   # (1, B*num_windows*C, patch_size, patch_size)
        # # shortcut = shortcut.reshape(1, B*num_windows_h*num_windows_w*C, self.patch_size, self.patch_size)
        # #
        # # shortcut = F.conv2d(
        # #     shortcut,
        # #     res_weight,
        # #     padding=1,
        # #     groups=B_win * C_win
        # # )
        # #
        # # # Reconstruct feature map
        # # # shortcut = shortcut.reshape(B*num_windows_h*num_windows_w*C, self.patch_size, self.patch_size)
        # # shortcut = rearrange(
        # #     shortcut,
        # #     '1 (b h w c) p1 p2 -> b c (h p1) (w p2)',
        # #     b=B,
        # #     h=num_windows_h,
        # #     w=num_windows_w,
        # #     p1=self.patch_size,
        # #     p2=self.patch_size
        # # )
        #
        # return spatial + channel + shortcut
        # ===== 3. Cross-Dimension Fusion =====
        # ===== 2. Channel Attention =====

        feats = [conv(x) for conv in self.convs]
        weights = self.fusion(torch.cat([
            F.adaptive_avg_pool2d(feat, 1) for feat in feats
        ], dim=1))
        channel_out = self.sigmoid(weights) * x

        attn_out = self.final_norm(
            (spatial + channel_out).permute(0,2,3,1)
        ).permute(0,3,1,2)

        # ===== 4. Enhanced Residual =====
        # residual_out = self.residual(shortcut)

        return attn_out + shortcut

def elsa_op(features, ghost_mul, ghost_add, h_attn, lam, gamma,
            kernel_size=5, dilation=1, stride=1, version=''):

    # lambda and gamma
    _B, _C = features.shape[:2]
    ks = kernel_size
    ghost_mul = ghost_mul ** lam if lam != 0 \
        else torch.ones(_B, _C, ks, ks, device=features.device, requires_grad=False)
    ghost_add = ghost_add * gamma if gamma != 0 \
        else torch.zeros(_B, _C, ks, ks, device=features.device, requires_grad=False)

    # if features.is_cuda and ghost_mul.is_cuda and ghost_add.is_cuda and h_attn.is_cuda:
    #     return elsa_function_cuda(features, ghost_mul, ghost_add, h_attn,
    #                               kernel_size, dilation, stride, version)
    # else:
    B, C, H, W = features.shape
    _pad = kernel_size // 2 * dilation
    features = F.unfold(
        features, kernel_size=kernel_size, dilation=dilation, padding=_pad, stride=stride) \
        .reshape(B, C, kernel_size ** 2, H * W)
    ghost_mul = ghost_mul.reshape(B, C, kernel_size ** 2, 1)
    ghost_add = ghost_add.reshape(B, C, kernel_size ** 2, 1)
    h_attn = h_attn.reshape(B, 1, kernel_size ** 2, H * W)
    filters = ghost_mul * h_attn + ghost_add  # B, C, K, N
    return (features * filters).sum(2).reshape(B, C, H, W)

class ChunkedCrossAttention(nn.Module):
    def __init__(self, feat_dim=512, num_chunks=8, head_dim=64, mask_ratio=0.0010):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_dim = feat_dim // num_chunks  # 每块维度
        assert feat_dim % num_chunks == 0

        # 用于QKV投影的轻量层
        self.to_q = nn.Linear(self.chunk_dim, head_dim)
        self.to_k = nn.Linear(self.chunk_dim, head_dim)
        self.to_v = nn.Linear(self.chunk_dim, head_dim)
        self.mask_ratio = mask_ratio

    def forward(self, x_s, x_t):
        # 分块: [B,512] -> [B, num_chunks, chunk_dim]
        x_s_chunks = x_s.view(-1, self.num_chunks, self.chunk_dim)
        x_t_chunks = x_t.view(-1, self.num_chunks, self.chunk_dim)

        # 计算Q,K,V (源域作Query, 目标域作Key/Value)
        q = self.to_q(x_s_chunks)  # [B, num_chunks, head_dim]
        k = self.to_k(x_t_chunks)  # [B, num_chunks, head_dim]
        v = self.to_v(x_t_chunks)  # [B, num_chunks, head_dim]

        # 交叉注意力得分
        attn = torch.einsum('bqd,bkd->bqk', q, k)  # [B, num_chunks, num_chunks]
        attn = F.softmax(attn / (self.chunk_dim ** 0.5), dim=-1)
        zeros = torch.zeros_like(attn)
        attn = torch.where(attn < self.mask_ratio, zeros, attn)

        # 加权聚合Value (捕获跨领域块间关系)
        out = torch.einsum('bqk,bkd->bqd', attn, v)  # [B, num_chunks, head_dim]

        # 压缩回特征维度: [B, num_chunks*head_dim]
        return out.reshape(x_s.shape[0], -1)

'''标准的'''
class MFN(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(MFN, self).__init__()

        # net = MixedFeatureNet.MixedFeatureNet()
        net = MixedFeatureNet()

        if pretrained:
            net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))

        self.features = nn.Sequential(*list(net.children())[:-4])
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_class)


    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        y = self.flatten(x)
        out = self.fc(y)
        # return out, x, head_out
        return out, y, None

class DDAMNet(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(DDAMNet, self).__init__()

        # net = MixedFeatureNet.MixedFeatureNet()
        net = MixedFeatureNet()

        if pretrained:
            net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
        # if pretrained:
        #     checkpoints = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
        #     keys = checkpoints.named_parameters()
        #     model_dict = net.state_dict()
        #     check = OrderedDict()
        #     # keys = deepcopy(checkpoints).keys()
        #     for key, param in keys:
        #         check[key] = param
        #         if key not in model_dict:
        #             # print(key)
        #             # del checkpoints[key]
        #             del check[key]
        #     # for key in keys:
        #     #     if key not in model_dict:
        #     #         # print(key)
        #     #         # del checkpoints[key]
        #     #         del check[key]
        #
        #     model_dict.update(check)
        #     net.load_state_dict(model_dict)

        self.features = nn.Sequential(*list(net.children())[:-4])
        self.num_head = num_head
        # for i in range(int(num_head)):
        #     setattr(self, "cat_head%d" % (i), CoordAttHead())
        # self.CA = CoTAttention(dim=512, kernel_size=3)
        # self.CA = Attention(dim=512, kernel_size=3)
        # self.CA = PC_Attention(dim=512, kernel_size=3)
        self.dsa = DCA(in_channels=512, n_directions=num_head)
        # self.CA = CoordAttHead()
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        # self.cla_do = nn.Sequential(
        #     nn.Linear(519, 1),
        #     nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        # heads = []
        #
        # for i in range(self.num_head):
        #     heads.append(getattr(self, "cat_head%d" % i)(x))
        # head_out = heads
        #
        # y = heads[0]
        # for i in range(1, self.num_head):
        #     y = torch.max(y, heads[i])
        # y = x * y
        y = self.dsa(x)
        # y = self.CA(x)
        y = self.Linear(y)
        y = self.flatten(y)
        out = self.fc(y)
        # return out, x, head_out
        return out, y, None



def pretrained_DCAMNF(args):
    model = DDAMNet(num_class=7, num_head=args.num_head, pretrained=False)

    '''parameter匹配时加载参数'''
    checkpoints = torch.load(args.pretrained, map_location='cuda:0')
    model.load_state_dict(checkpoints['model_state_dict'])

    return model

'''GCAM部分FER-VMamba: A robust facial expression recognition framework with global
compact attention and hierarchical feature interaction'''
# class DDAMNet(nn.Module):
#     def __init__(self, num_class=7, num_head=2, pretrained=True):
#         super(DDAMNet, self).__init__()
#
#         # net = MixedFeatureNet.MixedFeatureNet()
#         self.net = MixedFeatureNet()
#
#         # # if pretrained:
#         # #     net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
#         # if pretrained:
#         #     checkpoints = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
#         #     keys = checkpoints.named_parameters()
#         #     model_dict = net.state_dict()
#         #     check = OrderedDict()
#         #     # keys = deepcopy(checkpoints).keys()
#         #     for key, param in keys:
#         #         check[key] = param
#         #         if key not in model_dict:
#         #             # print(key)
#         #             # del checkpoints[key]
#         #             del check[key]
#         #     # for key in keys:
#         #     #     if key not in model_dict:
#         #     #         # print(key)
#         #     #         # del checkpoints[key]
#         #     #         del check[key]
#         #
#         #     model_dict.update(check)
#         #     net.load_state_dict(model_dict)
#         pretrained_state_dict = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
#         if hasattr(pretrained_state_dict, 'state_dict'):
#             state_dict_to_load = pretrained_state_dict.state_dict()
#         else:
#             state_dict_to_load = pretrained_state_dict
#         model_dict = self.net.state_dict()
#         pretrained_dict = {k: v for k, v in state_dict_to_load.items() if k in model_dict}
#         self.net.load_state_dict(pretrained_dict, strict=False)
#         # self.features = nn.Sequential(*list(net.children()))
#         # self.features = nn.Sequential(*list(net.children())[:-4])
#         # self.num_head = num_head
#         # for i in range(int(num_head)):
#         #     setattr(self, "cat_head%d" % (i), CoordAttHead())
#         # # self.CA = CoTAttention(dim=512, kernel_size=3)
#         # # self.CA = Attention(dim=512, kernel_size=3)
#         # # self.CA = PC_Attention(dim=512, kernel_size=3)
#         # # self.CA = CoordAttHead()
#         # self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
#         # self.flatten = Flatten()
#         # self.fc = nn.Linear(512, num_class)
#         # self.bn = nn.BatchNorm1d(num_class)
#         # # self.cla_do = nn.Sequential(
#         # #     nn.Linear(519, 1),
#         # #     nn.Sigmoid())
#
#     def forward(self, x):
#         x, out = self.net(x)
#         # # heads = []
#         # #
#         # # for i in range(self.num_head):
#         # #     heads.append(getattr(self, "cat_head%d" % i)(x))
#         # # head_out = heads
#         # #
#         # # y = heads[0]
#         # # for i in range(1, self.num_head):
#         # #     y = torch.max(y, heads[i])
#         # # y = x * y
#         #
#         # # y = self.CA(x)
#         # # y = self.Linear(x)
#         # y = self.flatten(x)
#         # out = self.fc(y)
#         # # return out, x, head_out
#
#         return out, x, None

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
                      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512, 512)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()

        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.Linear_h(x)  # (B, 512, 7, 1)
        x_w = self.Linear_w(x)  # (B, 512, 1, 7)
        x_w = x_w.permute(0, 1, 3, 2)   # (B, 512, 7, 1)

        y = torch.cat([x_h, x_w], dim=2)    # (B, 512, 14, 1)
        y = self.conv1(y)   # (B, 16, 14, 1)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)    # (B, 16, 7, 1)
        x_w = x_w.permute(0, 1, 3, 2)   # (B, 16, 1, 7)

        x_h = self.conv2(x_h).sigmoid()     # (B, 512, 7, 1)
        x_w = self.conv3(x_w).sigmoid()     # (B, 512, 1, 7)
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = x_w * x_h

        return y


# if __name__ == '__main__':
#     model = DDAMNet()
#     x = torch.randn((4, 3, 112, 112))
#     y = model(x)
#     a = 1

