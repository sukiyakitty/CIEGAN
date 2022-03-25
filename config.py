import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--input_scale_size', type=int, default=256,
                     help='input image will be resized with the given value as width and height')
net_arg.add_argument('--conv_hidden_num', type=int, default=128, choices=[128], help='default 128')
# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='CD09_Bright_256', choices=['Bright', 'DAPI', 'cTnT', 'Living', 'others'])
data_arg.add_argument('--train_path', type=str, default=r'E:\Data\CD09_Bright_256\AB\train')
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=8)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
data_arg.add_argument('--num_worker', type=int, default=4)
data_arg.add_argument('--gpu_number', type=str, default=0)
# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--using_loss_G_r', type=str2bool, default=True, help='Whether using reconstruction loss?')
train_arg.add_argument('--using_loss_G_p', type=str2bool, default=True, help='Whether using perceptual loss?')
train_arg.add_argument('--using_loss_G_a', type=str2bool, default=True, help='Whether using adversarial loss?')
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=800000)
train_arg.add_argument('--step_1', type=int, default=6000)
train_arg.add_argument('--step_2', type=int, default=7500)
train_arg.add_argument('--step_3', type=int, default=1000000)
train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
train_arg.add_argument('--d_lr', type=float, default=0.001)
train_arg.add_argument('--g_lr', type=float, default=0.001)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00002)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_r', type=float, default=1)
train_arg.add_argument('--lambda_p', type=float, default=1)
train_arg.add_argument('--lambda_a', type=float, default=0.02)
train_arg.add_argument('--lambda_c', type=float, default=0.05)
train_arg.add_argument('--lambda_s', type=float, default=50)
train_arg.add_argument('--lambda_t', type=float, default=0.1)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_step', type=int, default=1000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed