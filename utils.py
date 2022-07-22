import os
import math
import json
import logging
import numpy as np
import cv2
import datetime
from PIL import Image

_ = (256, 256)


def any_to_image(img):
    if type(img) is str:
        if not os.path.exists(img):
            print('!ERROR! The image path does not existed!')
            return None
        try:
            img = cv2.imread(img, cv2.IMREAD_COLOR)  # BGR .shape=(h,w,3)
        except:
            print('cv2.imread error! ')
        img = np.uint8(img)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif type(img) is np.ndarray:
        img = np.uint8(img)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = img[:, :, 0:3]
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            # img_gray = img
            print('!NOTICE! The input image is gray!')
            # pass
        else:
            print('!ERROR! The image shape error!')
            return None
    else:
        print('!ERROR! Please input correct CV2 image file or file path!')
        return None

    return img


def image_to_gray(img):
    img = any_to_image(img)
    if img is None:
        return None

    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        img_gray = img
    else:
        print('!ERROR! The image shape error!')
        return None

    return img_gray


def norm_img(image):
    image = image / 127.5 - 1
    return image


def trans_blur(img, ksize=13):
    if type(img) is np.ndarray:
        image = img
    else:
        print('!ERROR! Please input correct CV2 image!')
        return None

    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=0, sigmaY=0)


def trans_CLAHE(img, tileGridSize=16):
    if type(img) is np.ndarray:
        image = img
    else:
        print('!ERROR! Please input correct CV2 image!')
        return None

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(tileGridSize, tileGridSize))

    return clahe.apply(img)


def trans_Unsharp_Masking(img, ksize=13, k=1):
    if type(img) is np.ndarray:
        image = img
    else:
        print('!ERROR! Please input correct CV2 image!')
        return None

    img_blur = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=0, sigmaY=0)

    return np.uint8(img + k * (img - img_blur))


def trans_gamma(img, gamma=2.2):
    # input img is np.ndarray cv2 image file
    # gamma = 2.2
    # output np.ndarray
    if type(img) is np.ndarray:
        image = img
    else:
        print('!ERROR! Please input correct CV2 image!')
        return None

    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(image, gamma_table)


def trans_line(img, low_cut, hight_cut):
    # input img is np.ndarray cv2 image file
    # low_cut,hight_cut: line Histogram between: 0~1
    # output np.ndarray
    if type(img) is np.ndarray:
        image = img
    else:
        print('!ERROR! Please input correct CV2 image!')
        return None

    line_table = []
    for i in range(256):
        if i <= low_cut * 255:
            line_table.append(0)
        elif i >= hight_cut * 255:
            line_table.append(255)
        else:
            line_table.append((i - 255 * low_cut) / (hight_cut - low_cut))

    line_table = np.round(np.array(line_table)).astype(np.uint8)

    return cv2.LUT(image, line_table)


def save_traditional_enhancement(in_file, out_file):
    img_original = image_to_gray(in_file)
    img_enhanced = trans_CLAHE(trans_Unsharp_Masking(img_original))
    cv2.imwrite(out_file, img_enhanced)
    return True


def save_blur(in_file, out_file):
    img_original = image_to_gray(in_file)
    img_blur = trans_blur(img_original)
    cv2.imwrite(out_file, img_blur)
    return True


def image_resize(in_, out_, size=(256,256)):
    # resize all images in in_ and output in out_
    # in_ or out_ can be a image file path or dir path
    # if out_ is a dir, it must existed

    if os.path.isfile(in_):
        name_img = os.path.split(in_)[-1]
        o_img = any_to_image(in_)
        d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
        print(out_)
        if os.path.isdir(out_):
            cv2.imwrite(os.path.join(out_, name_img), d_img)
        else:
            cv2.imwrite(out_, d_img)
    elif os.path.isdir(in_):
        path_list = os.listdir(in_)
        for i in path_list:  # r'Label_1.png'
            img_dirfile = os.path.join(in_, i)
            image_resize(img_dirfile, out_, size=size)
    else:
        print('!ERROR! The input path or image does not existed!')
        return False

    return True


def folder_image_resize(image_path, size=(256, 256)):
    # resize all images in sub-folder and overwrite itself

    if not os.path.exists(image_path):
        print('!ERROR! The image_path does not existed!')
        return False

    path_list = os.listdir(image_path)

    for i in path_list:  # r'Label_1.png'
        img_dirfile = os.path.join(image_path, i)
        if os.path.isfile(img_dirfile):
            o_img = any_to_image(img_dirfile)
            d_img = cv2.resize(o_img, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(img_dirfile, d_img)
        else:
            folder_image_resize(img_dirfile, size=size)

    return True


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    print(config.model_name)
    print(config.model_dir)
    print(config.data_dir)
    print(config.load_path)


def get_time():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")
    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def rank(array):
    return len(array.shape)


def make_grid(tensor, nrow=8, padding=2, normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.ones([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h + h_width, w:w + w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
