import os
import shutil
import time
import math
import numpy as np
import scipy
import pandas as pd
import skimage.measure
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
import torch
import pyiqa
import lpips
from torch_fidelity import calculate_metrics
from niqe import getNIQE
from utils import any_to_image, image_to_gray, save_traditional_enhancement, save_blur

# import matlab.engine
try:
    import matlab.engine

    print('matlab.engine')
except:
    print('Importing matlab.engine fails. Please install Matlab.')


def call_test(in_file, out_file, ckp_dir):
    os.system(r'python test.py --input {} --output {} --checkpoint_dir {}'.format(in_file, out_file, ckp_dir))


def call_test_folder(in_dir, out_dir, ckp_dir):
    os.system(r'python test_folder.py --input {} --output {} --checkpoint_dir {}'.format(in_dir, out_dir, ckp_dir))


def test_folder(input_dir, output_dir, checkpoint_dir):
    # run CIEGAN, if big image then cut; if folder, image must be 128*128
    if not os.path.exists(input_dir):
        print('!ERROR! The input_dir path does not existed!')
        return False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_list = os.listdir(input_dir)
    for i in img_list:
        in_img_path = os.path.join(input_dir, i)
        # print(in_img_path)
        first_img = any_to_image(in_img_path)
        if first_img.shape[0] > 128 or first_img.shape[1] > 128:
            big_img = True
        else:
            big_img = False
        break

    if big_img:
        for i in img_list:
            in_img_path = os.path.join(input_dir, i)
            out_img_path = os.path.join(output_dir, i)
            call_test(in_img_path, out_img_path, checkpoint_dir)
    else:
        # if len(img_list) > 512:
        if len(img_list) > 512:
            img_list_split = [img_list[i:i + 512] for i in range(0, len(img_list), 512)]
        else:
            img_list_split = [img_list]

        for k, img_list_512 in enumerate(img_list_split):
            k_folder = os.path.join(input_dir, str(k))
            os.makedirs(k_folder)
            for img in img_list_512:
                from_file = os.path.join(input_dir, img)
                to_file = os.path.join(k_folder, img)
                shutil.copy(from_file, to_file)
            call_test_folder(k_folder, output_dir, checkpoint_dir)
            shutil.rmtree(k_folder)

    return True


def traditional_enhancement_method_folder(input_dir, output_dir):
    # EGT
    if not os.path.exists(input_dir):
        print('!ERROR! The input_dir path does not existed!')
        return False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_list = os.listdir(input_dir)
    for i in img_list:
        in_img_path = os.path.join(input_dir, i)
        out_img_path = os.path.join(output_dir, i)
        save_traditional_enhancement(in_img_path, out_img_path)
    return True


def traditional_blur_folder(input_dir, output_dir):
    # Gaussian Blur
    if not os.path.exists(input_dir):
        print('!ERROR! The input_dir path does not existed!')
        return False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_list = os.listdir(input_dir)
    for i in img_list:
        in_img_path = os.path.join(input_dir, i)
        out_img_path = os.path.join(output_dir, i)
        save_blur(in_img_path, out_img_path)
    return True


def evaluate_entropy_folder(main_path, in_dir):
    # once add all entropy to Evaluation.csv
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    col = 'Entropy_' + in_dir
    IN_dir = os.path.join(main_path, in_dir)
    if not os.path.exists(IN_dir):
        print('!ERROR! The Input_dir path does not existed!')
        return False

    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):  # existed!
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:  # not existed!
        Eval_df = pd.DataFrame(columns=col)
        Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    img_list = os.listdir(IN_dir)
    for i in img_list:
        in_img_path = os.path.join(IN_dir, i)
        img = image_to_gray(in_img_path)
        Eval_df.loc[i, 'Entropy_' + in_dir] = skimage.measure.shannon_entropy(img, base=2)

    Eval_df.to_csv(path_or_buf=Eval_file_path)

    return True


def evaluate_niqe_folder(main_path, in_dir):
    # BAD! once add all niqe to Evaluation.csv
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False

    input1 = os.path.join(main_path, in_dir)

    if not os.path.isdir(input1):
        print('!ERROR! The input must be a folder with images!')
        return False

    image_name_list = os.listdir(input1)
    for i, image in enumerate(image_name_list):

        i_img_1 = any_to_image(os.path.join(input1, image))
        i_img_1 = np.expand_dims(i_img_1, 0)
        # i_img_1 = np.expand_dims(i_img_1, -1)
        i_img_1 = np.transpose(i_img_1, (0, 3, 1, 2))

        if i == 0:
            images_1 = i_img_1
        else:
            images_1 = np.concatenate((images_1, i_img_1), axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    niqe_metric = pyiqa.create_metric('niqe').to(device)
    print('Is niqe the lower the better? ', niqe_metric.lower_better)

    # print(type(images_1))
    # print(images_1.shape)
    # print(images_1.dtype)
    images_1 = images_1.astype(np.float32)
    # print(type(images_1))
    # print(images_1.shape)
    # print(images_1.dtype)

    in_1 = torch.tensor(images_1).to(device)
    score_niqe = niqe_metric(in_1).detach().cpu().numpy()

    col = 'NIQE' + '_' + in_dir
    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:
        Eval_df = pd.DataFrame(columns=col)
        Eval_df.to_csv(path_or_buf=Eval_file_path)
    Eval_df.loc[image_name_list, col] = score_niqe
    Eval_df.to_csv(path_or_buf=Eval_file_path)

    return True


def evaluateLPIPSfolder(main_path, gt_dir, eval_dir):
    # once add all lpips to Evaluation.csv
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False

    input1 = os.path.join(main_path, eval_dir)
    input2 = os.path.join(main_path, gt_dir)

    if not os.path.isdir(input1) or not os.path.isdir(input2):
        print('!ERROR! The input must be a folder with images!')
        return False

    image_name_list = os.listdir(input1)
    for i, image in enumerate(image_name_list):

        i_img_1 = any_to_image(os.path.join(input1, image))
        i_img_1 = np.expand_dims(i_img_1, 0)
        # i_img_1 = np.expand_dims(i_img_1, -1)
        i_img_1 = np.transpose(i_img_1, (0, 3, 1, 2))

        i_img_2 = any_to_image(os.path.join(input2, image))
        i_img_2 = np.expand_dims(i_img_2, 0)
        # i_img_2 = np.expand_dims(i_img_2, -1)
        i_img_2 = np.transpose(i_img_2, (0, 3, 1, 2))

        if i == 0:
            images_1 = i_img_1
            images_2 = i_img_2
        else:
            images_1 = np.concatenate((images_1, i_img_1), axis=0)
            images_2 = np.concatenate((images_2, i_img_2), axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = pyiqa.create_metric('lpips').to(device)
    print('Is lpips the lower the better? ', lpips_metric.lower_better)

    in_1 = torch.tensor(images_1).to(device)
    in_2 = torch.tensor(images_2).to(device)
    score_lpips = lpips_metric(in_1, in_2).detach().cpu().numpy()

    col = 'LPIPS' + '_' + eval_dir + '~' + gt_dir
    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:
        Eval_df = pd.DataFrame(columns=col)
        Eval_df.to_csv(path_or_buf=Eval_file_path)
    Eval_df.loc[image_name_list, col] = score_lpips
    Eval_df.to_csv(path_or_buf=Eval_file_path)

    return True


def evaluateISFIDKIDfolder(main_path, gt_dir, eval_dir):
    # test IS FID KID (input must >=1000 images)
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False

    input1 = os.path.join(main_path, eval_dir)
    input2 = os.path.join(main_path, gt_dir)

    if not os.path.isdir(input1) or not os.path.isdir(input2):
        print('!ERROR! The input must be a folder with images!')
        return False

    # Each input can be either a string (path to images, registered input), or a Dataset instance
    # input1 or input2 must be >=1000 images folder
    metrics_dict = calculate_metrics(input1=input1, input2=input2, cuda=False, isc=True, fid=True, kid=True,
                                     verbose=False)

    print(metrics_dict)

    # Output:
    # {
    #     'inception_score_mean': 11.23678,
    #     'inception_score_std': 0.09514061,
    #     'frechet_inception_distance': 18.12198,
    #     'kernel_inception_distance_mean': 0.01369556,
    #     'kernel_inception_distance_std': 0.001310059
    # }

    Eval_file_path = os.path.join(main_path, 'Evaluation_Consolidation.csv')
    if os.path.exists(Eval_file_path):
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:
        Eval_df = pd.DataFrame()

    # cols=['IS'+'_' + eval_dir + '~' + gt_dir,'FID'+'_' + eval_dir + '~' + gt_dir, 'KID'+'_' + eval_dir + '~' + gt_dir]
    Eval_df.loc['Value', 'FID' + '_' + eval_dir + '~' + gt_dir] = metrics_dict['frechet_inception_distance']
    Eval_df.loc['Mean', 'IS' + '_' + eval_dir + '~' + gt_dir] = metrics_dict['inception_score_mean']
    Eval_df.loc['Std', 'IS' + '_' + eval_dir + '~' + gt_dir] = metrics_dict['inception_score_std']
    Eval_df.loc['Mean', 'KID' + '_' + eval_dir + '~' + gt_dir] = metrics_dict['kernel_inception_distance_mean']
    Eval_df.loc['Std', 'KID' + '_' + eval_dir + '~' + gt_dir] = metrics_dict['kernel_inception_distance_std']
    Eval_df.to_csv(path_or_buf=Eval_file_path)

    return metrics_dict


def evaluateFR_add_folder(main_path, gt_dir, eval_dir, add_name, add_funtion):
    # once add all add_funtion to Evaluation.csv
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False

    if add_funtion.__name__.find('folder') >= 0:
        add_funtion(main_path, gt_dir, eval_dir)
        return True
    elif add_funtion.__name__.find('get') >= 0:
        pass
    else:
        return False

    col = add_name + '_' + eval_dir + '~' + gt_dir

    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False
    EVAL_dir = os.path.join(main_path, eval_dir)
    if not os.path.exists(EVAL_dir):
        print('!ERROR! The CIEGAN_dir path does not existed!')
        return False

    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):  # existed!
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:  # not existed!
        Eval_df = pd.DataFrame()
        # Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    img_list = os.listdir(GT_dir)
    for i in img_list:
        GT_img_path = os.path.join(GT_dir, i)
        CIEGAN_img_path = os.path.join(EVAL_dir, i)

        # img_1 = image_to_gray(GT_img_path)
        # img_2 = image_to_gray(CIEGAN_img_path)

        t0 = time.time()
        Eval_df.loc[i, col] = add_funtion(GT_img_path, CIEGAN_img_path)
        t1 = time.time()
        time_cost = t1 - t0
        print("Image %s takes: %f sec" % (i, time_cost))
        if time_cost >= 3:
            Eval_df.to_csv(path_or_buf=Eval_file_path)

    Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    return True


def evaluateNR_add_folder(main_path, gt_dir, add_name, add_funtion):
    # once add all add_funtion to Evaluation.csv
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False

    if add_funtion.__name__.find('folder') >= 0:
        add_funtion(main_path, gt_dir)
        return True
    elif add_funtion.__name__.find('get') >= 0:
        pass
    else:
        return False

    col = add_name + '_' + gt_dir

    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False

    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):  # existed!
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:  # not existed!
        Eval_df = pd.DataFrame()
        # Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    img_list = os.listdir(GT_dir)
    for i in img_list:
        GT_img_path = os.path.join(GT_dir, i)

        # img_1 = image_to_gray(GT_img_path)

        t0 = time.time()
        Eval_df.loc[i, col] = add_funtion(GT_img_path)
        t1 = time.time()
        time_cost=t1 - t0
        print("Image %s takes: %f sec" % (i,time_cost))
        if time_cost>=3:
            Eval_df.to_csv(path_or_buf=Eval_file_path)

    Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    return True


def evaluate_folder(main_path, gt_dir, eval_dir):
    # once add all MSE RMSE NRMSE SSIM MS-SSIM PSNR UQI to Evaluation.csv
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False

    cols = ['MSE_' + eval_dir + '~' + gt_dir, 'RMSE_' + eval_dir + '~' + gt_dir, 'NRMSE_' + eval_dir + '~' + gt_dir,
            'SSIM_' + eval_dir + '~' + gt_dir, 'MS-SSIM_' + eval_dir + '~' + gt_dir, 'PSNR_' + eval_dir + '~' + gt_dir,
            'UQI_' + eval_dir + '~' + gt_dir]

    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False
    EVAL_dir = os.path.join(main_path, eval_dir)
    if not os.path.exists(EVAL_dir):
        print('!ERROR! The CIEGAN_dir path does not existed!')
        return False

    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):  # existed!
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:  # not existed!
        Eval_df = pd.DataFrame(columns=cols)
        Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    img_list = os.listdir(GT_dir)
    for i in img_list:
        GT_img_path = os.path.join(GT_dir, i)
        CIEGAN_img_path = os.path.join(EVAL_dir, i)

        t0 = time.time()
        Eval_df.loc[i, ['MSE_' + eval_dir + '~' + gt_dir, 'RMSE_' + eval_dir + '~' + gt_dir,
                        'NRMSE_' + eval_dir + '~' + gt_dir, 'SSIM_' + eval_dir + '~' + gt_dir,
                        'MS-SSIM_' + eval_dir + '~' + gt_dir, 'PSNR_' + eval_dir + '~' + gt_dir,
                        'UQI_' + eval_dir + '~' + gt_dir]] = evaluate_2images(GT_img_path, CIEGAN_img_path)
        t1 = time.time()
        time_cost = t1 - t0
        print("Image %s takes: %f sec" % (i, time_cost))
        if time_cost >= 3:
            Eval_df.to_csv(path_or_buf=Eval_file_path)

    Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    return True


def evaluate_folder_all(main_path, in_dir, GT_dir, CIEGAN_dir, TE_dir):
    # not use
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    cols = ['Entropy_' + in_dir, 'Entropy_' + GT_dir, 'Entropy_' + CIEGAN_dir, 'Entropy_' + TE_dir,
            'MSE_' + CIEGAN_dir + '~' + GT_dir, 'RMSE_' + CIEGAN_dir + '~' + GT_dir,
            'NRMSE_' + CIEGAN_dir + '~' + GT_dir, 'SSIM_' + CIEGAN_dir + '~' + GT_dir,
            'MS-SSIM_' + CIEGAN_dir + '~' + GT_dir, 'PSNR_' + CIEGAN_dir + '~' + GT_dir,
            'MSE_' + TE_dir + '~' + GT_dir, 'RMSE_' + TE_dir + '~' + GT_dir, 'NRMSE_' + TE_dir + '~' + GT_dir,
            'SSIM_' + TE_dir + '~' + GT_dir, 'MS-SSIM_' + TE_dir + '~' + GT_dir, 'PSNR_' + TE_dir + '~' + GT_dir]
    in_dir = os.path.join(main_path, in_dir)
    if not os.path.exists(in_dir):
        print('!ERROR! The Input_dir path does not existed!')
        return False
    GT_dir = os.path.join(main_path, GT_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False
    CIEGAN_dir = os.path.join(main_path, CIEGAN_dir)
    if not os.path.exists(CIEGAN_dir):
        print('!ERROR! The CIEGAN_dir path does not existed!')
        return False
    TE_dir = os.path.join(main_path, TE_dir)
    if not os.path.exists(TE_dir):
        print('!ERROR! The Traditional_Enhanced_dir path does not existed!')
        return False

    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):  # existed!
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:  # not existed!
        # print('The First time to processing Evaluation!')
        # initialize colums of Fractal.csv
        Eval_df = pd.DataFrame(columns=cols)
        Eval_df.to_csv(path_or_buf=Eval_file_path)  # save

    img_list = os.listdir(in_dir)
    for i in img_list:
        in_img_path = os.path.join(in_dir, i)
        GT_img_path = os.path.join(GT_dir, i)
        CIEGAN_img_path = os.path.join(CIEGAN_dir, i)
        TE_img_path = os.path.join(TE_dir, i)

        # evaluate_2images(GT_img_path, CIEGAN_img_path)
        # evaluate_2images(GT_img_path, TE_img_path)
        #
        # in_entropy = skimage.measure.shannon_entropy(in_img_path, base=2)
        # GT_entropy = skimage.measure.shannon_entropy(GT_img_path, base=2)
        # CIEGAN_entropy = skimage.measure.shannon_entropy(CIEGAN_img_path, base=2)
        # TE_entropy = skimage.measure.shannon_entropy(TE_img_path, base=2)

        Eval_df.loc[i, 'Entropy_' + in_dir] = skimage.measure.shannon_entropy(image_to_gray(in_img_path), base=2)
        Eval_df.loc[i, 'Entropy_' + GT_dir] = skimage.measure.shannon_entropy(image_to_gray(GT_img_path), base=2)
        Eval_df.loc[i, 'Entropy_' + CIEGAN_dir] = skimage.measure.shannon_entropy(image_to_gray(CIEGAN_img_path),
                                                                                  base=2)
        Eval_df.loc[i, 'Entropy_' + TE_dir] = skimage.measure.shannon_entropy(image_to_gray(TE_img_path), base=2)

        Eval_df.loc[i, ['MSE_' + CIEGAN_dir + '~' + GT_dir, 'RMSE_' + CIEGAN_dir + '~' + GT_dir,
                        'NRMSE_' + CIEGAN_dir + '~' + GT_dir, 'SSIM_' + CIEGAN_dir + '~' + GT_dir,
                        'MS-SSIM_' + CIEGAN_dir + '~' + GT_dir,
                        'PSNR_' + CIEGAN_dir + '~' + GT_dir]] = evaluate_2images(GT_img_path, CIEGAN_img_path)
        Eval_df.loc[
            i, ['MSE_' + TE_dir + '~' + GT_dir, 'RMSE_' + TE_dir + '~' + GT_dir, 'NRMSE_' + TE_dir + '~' + GT_dir,
                'SSIM_' + TE_dir + '~' + GT_dir, 'MS-SSIM_' + TE_dir + '~' + GT_dir,
                'PSNR_' + TE_dir + '~' + GT_dir]] = evaluate_2images(GT_img_path, TE_img_path)

    Eval_df.to_csv(path_or_buf=Eval_file_path)

    return True


def evaluate_2images(img_1, img_2):
    # MSE RMSE NRMSE SSIM MS-SSIM PSNR UQI
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    mse = skimage.measure.compare_mse(img_1, img_2)
    rmse = math.sqrt(mse)
    nrmse = skimage.measure.compare_nrmse(img_1, img_2, norm_type='Euclidean')

    ssim = skimage.measure.compare_ssim(img_1, img_2, data_range=255)
    msssim = MS_SSIM(img_1, img_2)

    psnr = skimage.measure.compare_psnr(img_1, img_2, data_range=255)
    uqi = getUQI(img_1, img_2)

    return [mse, rmse, nrmse, ssim, msssim, psnr, uqi]


def evaluate_2images_2(input1, input2):
    # test IS FID KID (input must >=1000 images)
    if not os.path.isdir(input1) or not os.path.isdir(input2):
        print('!ERROR! The input must be a folder with images!')
        return False
    # Each input can be either a string (path to images, registered input), or a Dataset instance
    # input1 or input2 must be >=1000 images folder
    metrics_dict = calculate_metrics(input1=input1, input2=input2, cuda=True, isc=True, fid=True, kid=True,
                                     verbose=False)

    print(metrics_dict)
    # Output:
    # {
    #     'inception_score_mean': 11.23678,
    #     'inception_score_std': 0.09514061,
    #     'frechet_inception_distance': 18.12198,
    #     'kernel_inception_distance_mean': 0.01369556,
    #     'kernel_inception_distance_std': 0.001310059
    # }

    return metrics_dict


def evaluate_2images_3(input1, input2):
    # test lpips niqe
    if not os.path.isdir(input1) or not os.path.isdir(input2):
        print('!ERROR! The input must be a folder with images!')
        return False

    image_name_list = os.listdir(input1)
    for i, image in enumerate(image_name_list):

        i_img_1 = any_to_image(os.path.join(input1, image))
        i_img_2 = any_to_image(os.path.join(input2, image))

        i_img_1 = np.expand_dims(i_img_1, 0)
        # i_img_1 = i_img_1[:, :, :, ::-1]
        i_img_1 = np.transpose(i_img_1, (0, 3, 1, 2))

        i_img_2 = np.expand_dims(i_img_2, 0)
        # i_img_2 = i_img_2[:, :, :, ::-1]
        i_img_2 = np.transpose(i_img_2, (0, 3, 1, 2))

        if i == 0:
            images_1 = i_img_1
            images_2 = i_img_2
        else:
            images_1 = np.concatenate((images_1, i_img_1), axis=0)
            images_2 = np.concatenate((images_2, i_img_2), axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # list all available metrics
    print(pyiqa.list_models())

    # create metric with default setting
    ssim_metric = pyiqa.create_metric('ssim').to(device)
    ms_ssim_metric = pyiqa.create_metric('ms_ssim').to(device)
    cw_ssim_metric = pyiqa.create_metric('cw_ssim').to(device)
    psnr_metric = pyiqa.create_metric('psnr').to(device)
    lpips_metric = pyiqa.create_metric('lpips').to(device)
    niqe_metric = pyiqa.create_metric('niqe').to(device)

    # create metric with custom setting
    # iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True).to(device)

    # check if lower better or higher better
    print('Is ssim the lower the better? ', ssim_metric.lower_better)
    print('Is ms_ssim the lower the better? ', ms_ssim_metric.lower_better)
    print('Is cw_ssim the lower the better? ', cw_ssim_metric.lower_better)
    print('Is psnr the lower the better? ', psnr_metric.lower_better)
    print('Is lpips the lower the better? ', lpips_metric.lower_better)
    print('Is niqe the lower the better? ', niqe_metric.lower_better)

    # print(type(images_1), images_1.shape)
    # print(type(images_2), images_2.shape)
    in_1 = torch.tensor(images_1).to(device)
    in_2 = torch.tensor(images_2).to(device)

    # example for iqa score inference
    # img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    # score_fr = iqa_metric(images_1, images_2)

    # score_ssim = ssim_metric(in_1, in_2).detach().cpu().numpy()
    # score_ms_ssim = ms_ssim_metric(in_1, in_2).detach().cpu().numpy()
    # score_cw_ssim = cw_ssim_metric(in_1, in_2).detach().cpu().numpy()
    # score_psnr = psnr_metric(in_1, in_2).cpu().detach().numpy()
    score_lpips = lpips_metric(in_1, in_2).detach().cpu().numpy()
    # score_niqe = niqe_metric(in_1, in_2).detach().cpu().numpy()

    # score_nr = iqa_metric(images_1)

    return score_lpips


def getSTD(img):
    img_gray = image_to_gray(img)
    (mean, stddv) = cv2.meanStdDev(img_gray)
    return stddv


def getEntropy(img):
    img_gray = image_to_gray(img)
    hist_256list = cv2.calcHist([img_gray], [0], None, [256], [0, 255])
    P_hist = hist_256list / (img_gray.shape[0] * img_gray.shape[1])
    entropy = -np.nansum(P_hist * np.log2(P_hist))
    return entropy


def getEntropy_(img):
    img_gray = image_to_gray(img)
    entropy = skimage.measure.shannon_entropy(img_gray, base=2)
    return entropy


def getAvgGradient(img):
    img_gray = image_to_gray(img)
    width = img_gray.shape[1]
    width = width - 1
    heigt = img_gray.shape[0]
    heigt = heigt - 1
    tmp = 0.0
    for i in range(width):
        for j in range(heigt):
            dx = float(img_gray[i, j + 1]) - float(img_gray[i, j])
            dy = float(img_gray[i + 1, j]) - float(img_gray[i, j])
            ds = math.sqrt((dx * dx + dy * dy) / 2)
            tmp += ds

    imageAG = tmp / (width * heigt)
    return imageAG


def getSpatialFrequency(img):
    img_gray = image_to_gray(img)
    M = img_gray.shape[0]
    N = img_gray.shape[1]
    cf = 0
    rf = 0
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(img_gray[i, j - 1]) - float(img_gray[i, j])
            rf += dx ** 2
            dy = float(img_gray[i - 1, j]) - float(img_gray[i, j])
            cf += dy ** 2
    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)
    return SF


def getFractal(img):
    matlab_engine = matlab.engine.start_matlab()
    fractal = matlab_engine.Task_Fractal_S(img)
    matlab_engine.quit()
    return fractal


def getResolution(img, pps=0.65):
    matlab_engine = matlab.engine.start_matlab()
    Resolution = matlab_engine.call_Dcorr(img, pps)
    matlab_engine.quit()
    return Resolution


def getMSE(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    mse = skimage.measure.compare_mse(img_1, img_2)
    return mse


def getRMSE(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    mse = skimage.measure.compare_mse(img_1, img_2)
    rmse = math.sqrt(mse)
    return rmse


def getNRMSE(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    nrmse = skimage.measure.compare_nrmse(img_1, img_2, norm_type='Euclidean')
    return nrmse


def getSSIM(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    ssim = skimage.measure.compare_ssim(img_1, img_2, data_range=255)
    return ssim


def getMSSSIM(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    msssim = MS_SSIM(img_1, img_2)
    return msssim


def getPSNR(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    psnr = skimage.measure.compare_psnr(img_1, img_2, data_range=255)
    return psnr


def getUQI(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    mean_1 = np.mean(img_1)
    mean_2 = np.mean(img_2)

    m_1, n_1 = np.shape(img_1)
    var_1 = np.sqrt(np.sum((img_1 - mean_1) ** 2) / (m_1 * n_1 - 1))
    m_2, n_2 = np.shape(img_2)
    var_2 = np.sqrt(np.sum((img_2 - mean_2) ** 2) / (m_2 * n_2 - 1))

    cov = np.sum((img_1 - mean_1) * (img_2 - mean_2)) / (m_1 * n_1 - 1)
    uqi = 4 * mean_1 * mean_2 * cov / ((mean_1 ** 2 + mean_2 ** 2) * (var_1 ** 2 + var_2 ** 2))
    return uqi


def getMI(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)
    return metrics.mutual_info_score(np.reshape(img_1, -1), np.reshape(img_2, -1))


def getNMI(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)
    return metrics.normalized_mutual_info_score(np.reshape(img_1, -1), np.reshape(img_2, -1))


def getAMI(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)
    return metrics.adjusted_mutual_info_score(np.reshape(img_1, -1), np.reshape(img_2, -1))


def getMI_(img_1, img_2):
    img_1 = image_to_gray(img_1)
    img_2 = image_to_gray(img_2)

    heigt, width = img_1.shape
    N = 256

    h = np.zeros((N, N))
    for i in range(heigt):
        for j in range(width):
            h[img_1[i, j], img_2[i, j]] = h[img_1[i, j], img_2[i, j]] + 1

    h = h / np.sum(h)

    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)

    H_x = 0
    H_y = 0

    for i in range(N):
        if (im1_marg[i] != 0):
            H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])

    for i in range(N):
        if (im2_marg[i] != 0):
            H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])

    H_xy = 0

    for i in range(N):
        for j in range(N):
            if (h[i, j] != 0):
                H_xy = H_xy + h[i, j] * math.log2(h[i, j])

    MI = H_xy - H_x - H_y

    return MI


class ImageEvalue(object):

    def image_mean(self, image):
        mean = np.mean(image)
        return mean

    def image_var(self, image, mean):
        m, n = np.shape(image)
        var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
        return var

    def images_cov(self, image1, image2, mean1, mean2):
        m, n = np.shape(image1)
        cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
        return cov

    def PSNR(self, O, F):
        MES = np.mean((np.array(O) - np.array(F)) ** 2)
        PSNR = 10 * np.log10(255 ** 2 / MES)
        return PSNR

    def SSIM(self, O, F):
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        meanO = self.image_mean(O)
        meanF = self.image_mean(F)
        varO = self.image_var(O, meanO)
        varF = self.image_var(O, meanF)
        covOF = self.images_cov(O, F, meanO, meanF)
        SSIM = (2 * meanO * meanF + c1) * (2 * covOF + c2) / (
                (meanO ** 2 + meanF ** 2 + c1) * (varO ** 2 + varF ** 2 + c2))
        return SSIM

    def IEF(self, O, F, X):
        IEF = np.sum((X - O) ** 2) / np.sum((F - O) ** 2)
        return IEF

    def UQI(self, O, F):
        meanO = self.image_mean(O)
        meanF = self.image_mean(F)
        varO = self.image_var(O, meanO)
        varF = self.image_var(F, meanF)
        covOF = self.images_cov(O, F, meanO, meanF)
        UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
        return UQI


def SSIM_index_new(img1, img2, K, win):
    # M, N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0] * 255) ** 2
    C2 = (K[1] * 255) ** 2
    win = win / np.sum(win)

    mu1 = scipy.signal.convolve2d(img1, win, mode='valid')
    mu2 = scipy.signal.convolve2d(img2, win, mode='valid')
    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1, img1), win, mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    # img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if (C1 > 0 and C2 > 0):
        # ssim1 = 2 * sigma12 + C2
        ssim_map = np.divide(np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)),
                             np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2)))
        cs_map = np.divide((2 * sigma12 + C2), (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1, denominator2)
        # 如果index是真，就赋值，是假就原值
        n, m = mu1.shape
        for i in range(n):
            for j in range(m):
                if (index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j] * numerator2[i][j] / denominator1[i][j] * denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if ((denominator1[i][j] != 0) and (denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j] / denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if (denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j] / denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]

    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return mssim, mcs


def MS_SSIM(img1, img2):
    K = [0.01, 0.03]
    win = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    # method = 'product'
    #
    # M, N = img1.shape
    # H, W = win.shape

    downsample_filter = np.ones((2, 2)) / 4
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mssim_array = []
    mcs_array = []

    for i in range(0, level):
        mssim, mcs = SSIM_index_new(img1, img2, K, win)
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2, ::2]
        img2 = filtered_im2[::2, ::2]

    # print(np.power(mcs_array[:level - 1], weight[:level - 1]))
    # print(mssim_array[level - 1] ** weight[level - 1])
    overall_mssim = np.prod(np.power(mcs_array[:level - 1], weight[:level - 1])) * (
            mssim_array[level - 1] ** weight[level - 1])

    return overall_mssim


def make_box_plot(main_path):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    Eval_file_path = os.path.join(main_path, 'Evaluation.csv')
    if os.path.exists(Eval_file_path):
        Eval_df = pd.read_csv(Eval_file_path, header=0, index_col=0)
        Eval_df = Eval_df.fillna(0)
        Eval_df = Eval_df.applymap(lambda x: float(x))
    else:
        return False
    figsize = (12.80, 10.24)
    fontsize = 12
    linewidth = 3
    folder = 2
    fontsize = fontsize * folder
    linewidth = linewidth * folder

    # NF_methods = ['STD', 'AG', 'SF', 'Entropy', 'NIQE', 'Fractal']
    NF_methods = ['STD', 'AG', 'SF', 'Entropy', 'NIQE', 'Resolution']
    cols = ['Input', 'GT', 'EGT', 'TE', 'CIEGAN', 'CIEGANP']
    for nf in NF_methods:
        this_box = Eval_df.boxplot(
            column=[nf + '_' + i for i in cols],
            figsize=(figsize[0] * folder, figsize[1] * folder), grid=True, fontsize=fontsize)
        plt.title(nf, fontsize=fontsize)
        plt.ylabel(nf, fontsize=fontsize)
        plt.xlabel(r'Conditions', fontsize=fontsize)
        plt.savefig(os.path.join(main_path, nf + r'.png'))
        plt.close()

    FR_methods = ['MSE', 'RMSE', 'NRMSE', 'SSIM', 'MS-SSIM', 'PSNR', 'UQI', 'LPIPS', 'MI', 'NMI', 'AMI']
    cols = ['TE~GT', 'CIEGAN~GT', 'EGT~GT', 'CIEGANP~GT', 'TE~EGT', 'CIEGAN~EGT']
    for fr in FR_methods:
        this_box = Eval_df.boxplot(
            column=[fr + '_' + i for i in cols],
            figsize=(figsize[0] * folder, figsize[1] * folder), grid=True, fontsize=fontsize)
        plt.title(fr, fontsize=fontsize)
        plt.ylabel(fr, fontsize=fontsize)
        plt.xlabel(r'Conditions', fontsize=fontsize)
        plt.savefig(os.path.join(main_path, fr + r'.png'))
        plt.close()

    return True


def test_bat(main_path, checkpoint_dir, gt_dir=r'GT'):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False

    blur_dir = r'Input'
    BLUR_dir = os.path.join(main_path, blur_dir)
    blur_enhance_dir = r'TE'
    BLUR_enhance_dir = os.path.join(main_path, blur_enhance_dir)
    gt_enhance_dir = r'EGT'
    GT_enhance_dir = os.path.join(main_path, gt_enhance_dir)
    ciegan_dir = r'CIEGAN'
    CIEGAN_dir = os.path.join(main_path, ciegan_dir)
    cieganp_dir = r'CIEGANP'
    CIEGANP_dir = os.path.join(main_path, cieganp_dir)

    traditional_blur_folder(GT_dir, BLUR_dir)
    traditional_enhancement_method_folder(GT_dir, GT_enhance_dir)
    traditional_enhancement_method_folder(BLUR_dir, BLUR_enhance_dir)
    test_folder(BLUR_dir, CIEGAN_dir, checkpoint_dir)
    test_folder(GT_dir, CIEGANP_dir, checkpoint_dir)

    # NR_name = ['STD', 'AG', 'SF', 'Entropy', 'NIQE', 'Fractal']
    NR_name = ['STD', 'AG', 'SF', 'Entropy', 'NIQE', 'Resolution']
    # NR_funtion = [getSTD, getAvgGradient, getSpatialFrequency, getEntropy, niqe, getFractal]
    NR_funtion = [getSTD, getAvgGradient, getSpatialFrequency, getEntropy, getNIQE, getResolution]
    cols = [blur_dir, gt_dir, gt_enhance_dir, ciegan_dir, blur_enhance_dir, cieganp_dir]
    for i in range(len(NR_name)):
        for dir in cols:
            evaluateNR_add_folder(main_path, dir, NR_name[i], NR_funtion[i])

    FR_name = ['MSE', 'RMSE', 'NRMSE', 'SSIM', 'MS-SSIM', 'PSNR', 'UQI', 'MI', 'NMI', 'AMI', 'LPIPS', 'ISFIDKID']
    FR_funtion = [getMSE, getRMSE, getNRMSE, getSSIM, getMSSSIM, getPSNR, getUQI, getMI, getNMI, getAMI,
                  evaluateLPIPSfolder, evaluateISFIDKIDfolder]
    cols = [gt_dir, gt_dir, gt_dir, gt_dir, gt_enhance_dir, gt_enhance_dir]
    cols_ref = [ciegan_dir, blur_enhance_dir, gt_enhance_dir, cieganp_dir, ciegan_dir, blur_enhance_dir]
    for i in range(len(FR_name)):
        for j in range(len(cols)):
            evaluateFR_add_folder(main_path, cols[j], cols_ref[j], FR_name[i], FR_funtion[i])

    make_box_plot(main_path)

    return True


def test_bat_1(main_path, checkpoint_dir, gt_dir=r'GT'):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False

    blur_dir = r'Input'
    BLUR_dir = os.path.join(main_path, blur_dir)
    blur_enhance_dir = r'TE'
    BLUR_enhance_dir = os.path.join(main_path, blur_enhance_dir)
    gt_enhance_dir = r'EGT'
    GT_enhance_dir = os.path.join(main_path, gt_enhance_dir)
    ciegan_dir = r'CIEGAN'
    CIEGAN_dir = os.path.join(main_path, ciegan_dir)
    cieganp_dir = r'CIEGANP'
    CIEGANP_dir = os.path.join(main_path, cieganp_dir)

    # test_folder(GT_dir, CIEGANP_dir, checkpoint_dir)

    NR_name = ['STD', 'AG', 'SF', 'Entropy', 'NIQE']
    NR_funtion = [getSTD, getAvgGradient, getSpatialFrequency, getEntropy, getNIQE]
    cols = [cieganp_dir]
    for i in range(len(NR_name)):
        for dir in cols:
            evaluateNR_add_folder(main_path, dir, NR_name[i], NR_funtion[i])

    FR_name = ['MSE', 'RMSE', 'NRMSE', 'SSIM', 'MS-SSIM', 'PSNR', 'UQI', 'MI', 'NMI', 'AMI', 'LPIPS']
    FR_funtion = [getMSE, getRMSE, getNRMSE, getSSIM, getMSSSIM, getPSNR, getUQI, getMI, getNMI, getAMI,
                  evaluateLPIPSfolder]
    cols = [gt_dir]
    cols_ref = [cieganp_dir]
    for i in range(len(FR_name)):
        for j in range(len(cols)):
            evaluateFR_add_folder(main_path, cols[j], cols_ref[j], FR_name[i], FR_funtion[i])

    # traditional_blur_folder(GT_dir, BLUR_dir)
    # traditional_enhancement_method_folder(GT_dir, GT_enhance_dir)
    # traditional_enhancement_method_folder(BLUR_dir, BLUR_enhance_dir)
    # test_folder(BLUR_dir, CIEGAN_dir, checkpoint_dir)
    #
    # evaluate_entropy_folder(main_path, blur_dir)
    # evaluate_entropy_folder(main_path, gt_dir)
    # evaluate_entropy_folder(main_path, gt_enhance_dir)
    # evaluate_entropy_folder(main_path, ciegan_dir)
    # evaluate_entropy_folder(main_path, blur_enhance_dir)
    #
    # evaluate_folder(main_path, gt_dir, ciegan_dir)
    # evaluate_folder(main_path, gt_dir, blur_enhance_dir)
    # evaluate_folder(main_path, gt_dir, gt_enhance_dir)
    # evaluate_folder(main_path, gt_enhance_dir, ciegan_dir)
    # evaluate_folder(main_path, gt_enhance_dir, blur_enhance_dir)

    # evaluate_lpips_folder(main_path, gt_dir, ciegan_dir)
    # evaluate_lpips_folder(main_path, gt_dir, blur_enhance_dir)
    # evaluate_lpips_folder(main_path, gt_dir, gt_enhance_dir)
    # evaluate_lpips_folder(main_path, gt_enhance_dir, ciegan_dir)
    # evaluate_lpips_folder(main_path, gt_enhance_dir, blur_enhance_dir)

    # evaluateNR_add_folder(main_path, blur_dir, 'NIQE', niqe)
    # evaluateNR_add_folder(main_path, gt_dir, 'NIQE', niqe)
    # evaluateNR_add_folder(main_path, gt_enhance_dir, 'NIQE', niqe)
    # evaluateNR_add_folder(main_path, ciegan_dir, 'NIQE', niqe)
    # evaluateNR_add_folder(main_path, blur_enhance_dir, 'NIQE', niqe)

    # evaluateNR_add_folder(main_path, blur_dir, 'Entropy', getEntropy)
    # evaluateNR_add_folder(main_path, gt_dir, 'Entropy', getEntropy)
    # evaluateNR_add_folder(main_path, gt_enhance_dir, 'Entropy', getEntropy)
    # evaluateNR_add_folder(main_path, ciegan_dir, 'Entropy', getEntropy)
    # evaluateNR_add_folder(main_path, blur_enhance_dir, 'Entropy', getEntropy)

    # evaluateNR_add_folder(main_path, blur_dir, 'Fractal', getFractal)
    # evaluateNR_add_folder(main_path, gt_dir, 'Fractal', getFractal)
    # evaluateNR_add_folder(main_path, gt_enhance_dir, 'Fractal', getFractal)
    # evaluateNR_add_folder(main_path, ciegan_dir, 'Fractal', getFractal)
    # evaluateNR_add_folder(main_path, blur_enhance_dir, 'Fractal', getFractal)

    # evaluateNR_add_folder(main_path, blur_dir, 'AG', getAvgGradient)
    # evaluateNR_add_folder(main_path, gt_dir, 'AG', getAvgGradient)
    # evaluateNR_add_folder(main_path, gt_enhance_dir, 'AG', getAvgGradient)
    # evaluateNR_add_folder(main_path, ciegan_dir, 'AG', getAvgGradient)
    # evaluateNR_add_folder(main_path, blur_enhance_dir, 'AG', getAvgGradient)
    #
    # evaluateNR_add_folder(main_path, blur_dir, 'SF', getSpatialFrequency)
    # evaluateNR_add_folder(main_path, gt_dir, 'SF', getSpatialFrequency)
    # evaluateNR_add_folder(main_path, gt_enhance_dir, 'SF', getSpatialFrequency)
    # evaluateNR_add_folder(main_path, ciegan_dir, 'SF', getSpatialFrequency)
    # evaluateNR_add_folder(main_path, blur_enhance_dir, 'SF', getSpatialFrequency)
    #
    # evaluateNR_add_folder(main_path, blur_dir, 'STD', getSTD)
    # evaluateNR_add_folder(main_path, gt_dir, 'STD', getSTD)
    # evaluateNR_add_folder(main_path, gt_enhance_dir, 'STD', getSTD)
    # evaluateNR_add_folder(main_path, ciegan_dir, 'STD', getSTD)
    # evaluateNR_add_folder(main_path, blur_enhance_dir, 'STD', getSTD)

    # evaluateFR_add_folder(main_path, gt_dir, ciegan_dir, 'MI', getMI)
    # evaluateFR_add_folder(main_path, gt_dir, blur_enhance_dir, 'MI', getMI)
    # evaluateFR_add_folder(main_path, gt_dir, gt_enhance_dir, 'MI', getMI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, ciegan_dir, 'MI', getMI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, blur_enhance_dir, 'MI', getMI)
    #
    # evaluateFR_add_folder(main_path, gt_dir, ciegan_dir, 'NMI', getNMI)
    # evaluateFR_add_folder(main_path, gt_dir, blur_enhance_dir, 'NMI', getNMI)
    # evaluateFR_add_folder(main_path, gt_dir, gt_enhance_dir, 'NMI', getNMI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, ciegan_dir, 'NMI', getNMI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, blur_enhance_dir, 'NMI', getNMI)
    #
    # evaluateFR_add_folder(main_path, gt_dir, ciegan_dir, 'AMI', getAMI)
    # evaluateFR_add_folder(main_path, gt_dir, blur_enhance_dir, 'AMI', getAMI)
    # evaluateFR_add_folder(main_path, gt_dir, gt_enhance_dir, 'AMI', getAMI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, ciegan_dir, 'AMI', getAMI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, blur_enhance_dir, 'AMI', getAMI)

    # -------------------------------------------------------------

    # evaluateFR_add_folder(main_path, gt_dir, ciegan_dir, 'UQI', UQI)
    # evaluateFR_add_folder(main_path, gt_dir, blur_enhance_dir, 'UQI', UQI)
    # evaluateFR_add_folder(main_path, gt_dir, gt_enhance_dir, 'UQI', UQI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, ciegan_dir, 'UQI', UQI)
    # evaluateFR_add_folder(main_path, gt_enhance_dir, blur_enhance_dir, 'UQI', UQI)

    # evaluate_niqe_folder(main_path, blur_dir)
    # evaluate_niqe_folder(main_path, gt_dir)
    # evaluate_niqe_folder(main_path, gt_enhance_dir)
    # evaluate_niqe_folder(main_path, ciegan_dir)
    # evaluate_niqe_folder(main_path, blur_enhance_dir)

    make_box_plot(main_path)

    return True


def test_bat_2(main_path, checkpoint_dir, gt_dir=r'GT'):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False

    blur_dir = r'Input'
    BLUR_dir = os.path.join(main_path, blur_dir)
    blur_enhance_dir = r'TE'
    BLUR_enhance_dir = os.path.join(main_path, blur_enhance_dir)
    gt_enhance_dir = r'EGT'
    GT_enhance_dir = os.path.join(main_path, gt_enhance_dir)
    ciegan_dir = r'CIEGAN'
    CIEGAN_dir = os.path.join(main_path, ciegan_dir)
    cieganp_dir = r'CIEGANP'
    CIEGANP_dir = os.path.join(main_path, cieganp_dir)

    FR_name = ['ISFIDKID']
    FR_funtion = [evaluateISFIDKIDfolder]
    cols = [gt_dir, gt_dir, gt_dir, gt_dir, gt_enhance_dir, gt_enhance_dir]
    cols_ref = [ciegan_dir, blur_enhance_dir, gt_enhance_dir, cieganp_dir, ciegan_dir, blur_enhance_dir]
    for i in range(len(FR_name)):
        for j in range(len(cols)):
            evaluateFR_add_folder(main_path, cols[j], cols_ref[j], FR_name[i], FR_funtion[i])


def test_bat_3(main_path, checkpoint_dir, gt_dir=r'GT'):
    if not os.path.exists(main_path):
        print('!ERROR! The main_path path does not existed!')
        return False
    GT_dir = os.path.join(main_path, gt_dir)
    if not os.path.exists(GT_dir):
        print('!ERROR! The Ground_Truth_dir path does not existed!')
        return False

    blur_dir = r'Input'
    BLUR_dir = os.path.join(main_path, blur_dir)
    blur_enhance_dir = r'TE'
    BLUR_enhance_dir = os.path.join(main_path, blur_enhance_dir)
    gt_enhance_dir = r'EGT'
    GT_enhance_dir = os.path.join(main_path, gt_enhance_dir)
    ciegan_dir = r'CIEGAN'
    CIEGAN_dir = os.path.join(main_path, ciegan_dir)
    cieganp_dir = r'CIEGANP'
    CIEGANP_dir = os.path.join(main_path, cieganp_dir)

    NR_name = ['Resolution']
    NR_funtion = [getResolution]
    cols = [blur_dir, gt_dir, gt_enhance_dir, ciegan_dir, blur_enhance_dir, cieganp_dir]
    for i in range(len(NR_name)):
        for dir in cols:
            evaluateNR_add_folder(main_path, dir, NR_name[i], NR_funtion[i])
    make_box_plot(main_path)


if __name__ == "__main__":
    # img = r'C:\DATA\CIEGAN_eval_5\GT\2018-09-13~F_CD09~T1_47.png'
    # resolution = getResolution(img, pps=0.65)
    # print(resolution)

    main_path = r'C:\DATA\CIEGAN_eval_4'
    checkpoint_dir = r'C:\DATA\PSL_CKP\checkpoint_20220304_235728'
    test_bat_3(main_path, checkpoint_dir, gt_dir=r'GT')

    # evaluate_niqe_folder(main_path, r'Input')
    # evaluate_niqe_folder(main_path, r'TE')
    # evaluate_niqe_folder(main_path, r'GT')
    # evaluate_niqe_folder(main_path, r'EGT')
    # evaluate_niqe_folder(main_path, r'CIEGAN')
    # evaluate_lpips_folder(main_path, r'GT', r'CIEGAN')
    # evaluate_lpips_folder(main_path, r'GT', r'TE')
    # evaluate_lpips_folder(main_path, r'GT', r'EGT')
    # evaluate_lpips_folder(main_path, r'EGT', r'CIEGAN')
    # evaluate_lpips_folder(main_path, r'EGT', r'TE')

    # input1 = r'C:\DATA\CIEGAN_eval_3\EGT'
    # input2 = r'C:\DATA\CIEGAN_eval_3\CIEGAN'
    # score = evaluate_2images_3(input1, input2)
    # print(score)
    # main_path = r'C:\DATA\CIEGAN_eval_2'
    # checkpoint_dir = r'C:\DATA\PSL_CKP\checkpoint_20220305_232233'
    # test_bat(main_path, checkpoint_dir, gt_dir=r'GT')
