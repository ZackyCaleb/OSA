import torch
import torch.nn as nn

class JointMIEstimator(nn.Module):
    def __init__(self, f_dim=512, y_dim=7, hidden_dim=256):
        super().__init__()
        self.y_dim = y_dim

        # 网络结构保持不变
        self.joint_net = nn.Sequential(
            nn.Linear(f_dim * 2 + y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1))

        self.marginal_net = nn.Sequential(
            nn.Linear(f_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, fs, ft, y):
        # 1. 计算 p(y) 的经验分布（更安全的实现）
        y_onehot = y.float()
        p_y = y_onehot.mean(dim=0, keepdim=True).clamp(min=1e-6, max=1.0)
        log_p_y = torch.log(p_y)

        # 计算网络输出并限制范围
        log_p_joint = self.joint_net(torch.cat([fs, ft, y_onehot], 1))
        log_p_marginal = self.marginal_net(torch.cat([fs, ft], 1))

        # 限制数值范围
        log_p_joint = torch.clamp(log_p_joint, min=-50, max=50)
        log_p_marginal = torch.clamp(log_p_marginal, min=-50, max=50)

        # 计算log密度比
        log_ratio = 1 + log_p_joint - log_p_marginal - log_p_y
        log_ratio = torch.clamp(log_ratio, min=-20, max=20)  # 进一步限制范围

        return log_ratio

        # 方案B：使用 scaled_exp 避免爆炸
        # scale = torch.max(log_ratio.detach(), dim=1, keepdim=True)[0]
        # g = torch.exp(log_ratio - scale) + 1e-10
        # return g * torch.exp(scale)

def get_class_aligned_features(f_s, f_t, y_s, y_t):
    # 筛选高置信度目标域样本
    # conf, y_t = torch.max(y_t_pred, dim=1)
    # mask = conf > confidence_threshold
    # f_t = f_t[mask]
    # y_t = y_t[mask]

    # 找到共有的类别
    # common_classes = torch.unique(y_s[y_s.isin(y_t)])
    # common_classes = y_s[(y_s.unsqueeze(1) == y_t).any(dim=1)]
    common_mask = torch.zeros_like(y_s, dtype=torch.bool)
    for label in y_t:
        common_mask = common_mask | (y_s == label)
    common_classes = torch.unique(y_s[common_mask])

    # 按类别对齐
    f_s_list, f_t_list, y_list = [], [], []

    for cls in common_classes:
        # 同类样本索引
        idx_s = (y_s == cls)
        idx_t = (y_t == cls)

        # 确保每类至少有1个样本
        if idx_s.sum() > 0 and idx_t.sum() > 0:
            # 重复采样使数量匹配
            num_pairs = min(idx_s.sum(), idx_t.sum())
            f_s_cls = f_s[idx_s][:num_pairs]
            f_t_cls = f_t[idx_t][:num_pairs]

            f_s_list.append(f_s_cls)
            f_t_list.append(f_t_cls)
            y_list.append(torch.full((num_pairs,), cls, device=f_s.device))

    return torch.cat(f_s_list), torch.cat(f_t_list), torch.cat(y_list)

def mi_loss_mlp(f_s, f_t, y_s, y_t_fake, mi_estimator):
    # 正样本对 (f_s, f_t, y)
    f_s, f_t, y = get_class_aligned_features(f_s, f_t, y_s, y_t_fake)
    y_onehot = torch.zeros(len(y), 7).cuda()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)  # 转one-hot
    pos = mi_estimator(f_s, f_t, y_onehot)
    # 负样本对 (f_s, f_t, y_shuffled)
    y_shuffled = y_onehot[torch.randperm(y_onehot.size(0))]
    # neg = torch.exp(mi_estimator(f_s, f_t, y_shuffled) - 1)
    neg = torch.exp(torch.clamp(mi_estimator(f_s, f_t, y_shuffled) - 1, min=-10, max=10))  # 截断
    # c_loss = center_loss(f_s, f_t, y)
    # js_loss = js_divergence_loss(f_s, f_t, y)
    # kl_loss = kl_divergence_loss(f_s, f_t, y)
    # return -(pos.mean() - neg.mean()) + 0.1*c_loss
    return -(pos.mean() - neg.mean())

    # t_onehot = torch.zeros(len(y), 7, device=y.device)
    # t_onehot.scatter_(1, y.unsqueeze(1), 1)  # (n_samples, n_classes)
    # t_counts = t_onehot.sum(dim=0)
    # center_t = (t_onehot.T @ t_logit) / (t_counts.unsqueeze(1) + 1e-6)
    #
    # # softmax_out = nn.Softmax(dim=1)(center_t)
    # L_cond = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-5), dim=1))
    # # msoftmax = softmax_out.mean(dim=0)
    # # L_marg = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    # # js_loss = 0.1 * L_cond - 0.1 * L_marg
    # # js_loss = - 0.1 * L_marg
    # js_loss = 0.1 * L_cond
    # return -(pos.mean() - neg.mean())



class L_je:
    # def __init__(self, lambda_1_init, lambda_1_max, lambda_2_max, lambda_2_final, T):
    def __init__(self, lambda_1, lambda_2):
        super(L_je, self).__init__()
        # self.lambda_1_init = lambda_1_init  # lambda_1起始值
        # self.lambda_1_max = lambda_1_max  # lambda_1最终值
        # self.lambda_2_max = lambda_2_max  # lambda_2起始值
        # self.lambda_2_final = lambda_2_final  # lambda_2最终值
        # self.T = T
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    '''原来的'''
    # def l1_linear(self, t):
    #     # lambda_tmp = self.lambda_max + 0.5 * (self.lambda_min-self.lambda_max) * (1 + math.cos(math.pi * t / self.T))
    #     return self.lambda_1_init + (self.lambda_1_max - self.lambda_1_init) * (t / self.T)
    #
    #
    # def l2_linear(self, t):
    #     # lambda_tmp = self.lambda_min + 0.5 * (self.lambda_max-self.lambda_min) * (1 + math.cos(math.pi * t / self.T))
    #     return self.lambda_2_max + (self.lambda_2_final - self.lambda_2_max) * (t / self.T)

    # def estimated(self, target_probs, t):
    def estimated(self, target_probs):
        # prob = torch.softmax(target_probs, dim=1)
        # 熵最小化项: 每个样本的预测熵
        softmax_out = nn.Softmax(dim=1)(target_probs)
        L_cond = torch.mean(torch.sum(-softmax_out * torch.log(softmax_out + 1e-5), dim=1))
        # lambda_1 = self.l1_linear(t)
        # lambda_1 = 0.1
        # 多样性最大化项: 类别边际分布的熵
        msoftmax = softmax_out.mean(dim=0)
        L_marg = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
        # lambda_2 = self.l2_linear(t)
        # lambda_2 = 0.1
        # return self.lambda_1 * L_cond - self.lambda_2 * L_marg  # 注意符号方向
        # return self.lambda_1 * L_cond
        return -self.lambda_2 * L_marg


def entropy_loss(predictions):
    prob = torch.softmax(predictions, dim=1)
    log_prob = torch.log_softmax(predictions, dim=1)
    entropy = -torch.sum(prob * log_prob, dim=1).mean()  # 平均熵
    return entropy



def safe_gather(log_q, y, num_classes):
    # 确保y在合法范围内

    y_clamped = torch.clamp(y, 0, num_classes - 1)

    # 过滤无效样本（可选）
    valid_mask = (y == y_clamped)
    if not valid_mask.all():
        print(f"Warning: {len(y) - valid_mask.sum()} invalid labels filtered")

    # 安全gather
    return log_q[valid_mask].gather(1, y_clamped[valid_mask].unsqueeze(1))
