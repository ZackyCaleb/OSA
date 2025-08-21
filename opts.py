import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# General options
# parser.add_argument('-shuffle', action='store_true', help='Reshuffle data at each epoch')
parser.add_argument('-shuffle', default=True, help='Reshuffle data at each epoch')
parser.add_argument('-train_record', action='store_true', help='Path to save train record')
parser.add_argument('-save_best_model_only', action='store_true', help='only save the best model')
parser.add_argument('-save_every_model', action='store_true', help='only save all models')
parser.add_argument('-test_only', action='store_true', help='Only conduct test on the validation set')
# parser.add_argument('-aug', action='store_true', help='augmentation the training images')
parser.add_argument('-aug', default=True, help='augmentation the training images')

# parser.add_argument('-model', required=True, help='Model type when we create a new one')
parser.add_argument('-model', default='DDAMFN', help='Model type when we create a new one')

parser.add_argument('-pretrained', default=r'./pretrained/MFN_msceleb.pth', help='Model type when we create a new one')


parser.add_argument('-train_list0', default=r'./data/raf/train', help='Path to rafdb2.0 ')
# parser.add_argument('-train_list0', default=r'./data/affect/train', help='Path to rafdb2.0 ')
# parser.add_argument('-train_list0', default=r'./data/fer2013/train', help='Path to rafdb2.0 ')
parser.add_argument('-train_list1', required=False, help='Path to rafdb')
# parser.add_argument('-test_list1', required=True, help='Path to data directory')
parser.add_argument('-test_list2', required=False, help='Path to data directory')
parser.add_argument('-test_list3', required=False, help='Path to data directory')
parser.add_argument('-test_list4', required=False, help='Path to data directory')
parser.add_argument('-test_list5', required=False, help='Path to data directory')
# parser.add_argument('-test_list6', required=False, help='Path to data directory')
parser.add_argument('-test_list6', default=r'./data/raf/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/sfew/retina/train', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/kdef/gen_test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/affect/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/mmi/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/fer2013/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/ck+/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/kdef/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/oulu/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/expw/test', help='Path to data directory')
# parser.add_argument('-test_list6', default=r'./data/sfew/retina/valid', help='Path to data directory')
parser.add_argument('-test_list7', required=False, help='Path to data directory')
parser.add_argument('-test_list8', required=False, help='Path to data directory')
parser.add_argument('-test_list9', required=False, help='Path to data directory')
# parser.add_argument('-test_list10', required=False, help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/fer2013/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/raf/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/ck+/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/kdef/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/affect/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/mmi/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/expw/train', help='Path to data directory')
# parser.add_argument('-test_list10', default=r'./data/sfew/train', help='Path to data directory')
# parser.add_argument('-print', required=True, help='information of the training hyper parameters')z
parser.add_argument('-print', default=True, help='information of the training hyper parameters')
# parser.add_argument('-target', default=None, required=False,type=str, help='target domain')
parser.add_argument('-target', default='jaf', required=False, type=str, help='target domain')
parser.add_argument('-get_features', default='source', required=False, type=str, help='get source or target features')
# parser.add_argument('-train_data', required=True, help='training data')
parser.add_argument('-train_data', default='raf2', help='training data')
# parser.add_argument('-test_data', required=True, help='testing data')
parser.add_argument('-test_data', default='jaf', help='testing data')
# parser.add_argument('-save_path', required=True, help='train:the dir to save train record,'
# parser.add_argument('-save_path', default=r'./SAFN',
# parser.add_argument('-save_path', default=r'./raf_check',
# parser.add_argument('-save_path', default=r'./fer_check',
# parser.add_argument('-save_path', default=r'./sfew_check',
# parser.add_argument('-save_path', default=r'./expw_check',
# parser.add_argument('-save_path', default=r'./affect_check',
# parser.add_argument('-save_path', default=r'./mmi_check',
# parser.add_argument('-save_path', default=r'./kdef_check',
parser.add_argument('-save_path', default=r'./ck_check',
# parser.add_argument('-save_path', default=r'./fer_check',
                    help='train:the dir to save train record, test_only: the model pth file path')
# parser.add_argument('-log_path', required=True, help='path to save csv file')
parser.add_argument('-log_path', default=r'./SAFN_log', help='path to save csv file')
# parser.add_argument('-output_classes', required=True, type=int, help='Num of emo classes')
parser.add_argument('-output_classes', default=7, type=int, help='Num of emo classes')

# Training options
parser.add_argument('-learn_rate', default=1e-4, type=float, help='Base learning rate of training')
parser.add_argument('-momentum', default=0.9, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=0.0005, type=float, help='Weight decay for training')
# parser.add_argument('-lam1', default=1.0, type=float, help='Weight of cross domain loss')
# parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-n_epochs', default=60, type=int, help='Training epochs')
parser.add_argument('-batch_size', default=128, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-criterion', default='mc_loss_center', help='focal_loss,my_loss,or none')
parser.add_argument('-opti', default='Adam', help='optimizer,SGD,Adam')
parser.add_argument('--lambda_1', default=1.0, type=float, help='trade-off parameter')
parser.add_argument('--lambda_2', default=1.0, type=float, help='trade-off parameter')

# Model options
parser.add_argument('-resume', action='store_true', help='Whether continue to train from a previous checkpoint')
parser.add_argument('-nGPU', default="0,1,2,3",type=str, help='which GPUs for training')
# parser.add_argument('-nGPU', default="5", type=str, help='which GPUs for training')
parser.add_argument('-workers', default=4, type=int, help='Number of subprocesses to to load data')
# parser.add_argument('-decay', default=8, type=int, help='LR decay')
parser.add_argument('-decay', default=30, type=int, help='LR decay')
parser.add_argument('-size', default=112, type=int)
# parser.add_argument('-num_classes', default=7, type=int)


args = parser.parse_args()
