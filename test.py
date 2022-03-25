import os
import sys
import argparse
import math
import cv2
import numpy as np
from models import denorm_img, build_server_graph
from utils import any_to_image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=r'C:\DATA\test\test_20.png', type=str,
                        help='The input image file.')
    parser.add_argument('--output', default=r'C:\DATA\test\test_output_7.png', type=str, help='The output image file.')
    parser.add_argument('--checkpoint_dir', default=r'C:\DATA\PSL_CKP\CKP_20220303_0', type=str,
                        help='Checkpoint directory.')
    parser.add_argument('--input_scale_size', type=int, default=128)
    parser.add_argument('--conv_hidden_num', type=int, default=128, choices=[128], help='default 128')
    return parser.parse_args(argv)


def single_image(args):
    block_size = args.input_scale_size
    o_img = any_to_image(args.input)
    if o_img is None:
        return False
    height = o_img.shape[0]
    width = o_img.shape[1]
    block_height = math.ceil(height / block_size)
    block_width = math.ceil(width / block_size)

    need_resize = False
    if block_height * block_size != height or block_width * block_size != width:
        need_resize = True
        big_img = cv2.resize(o_img, (block_width * block_size, block_height * block_size),
                             interpolation=cv2.INTER_NEAREST)  # (col width, row height)
    else:
        big_img = o_img

    box_list = []  # (col0, row0, col1, row1)
    for i in range(0, block_height):  # row height
        for j in range(0, block_width):  # col width
            box = (i * block_size, j * block_size, (i + 1) * block_size, (j + 1) * block_size)
            box_list.append(box)
    image_list = [big_img[box[1]:box[3], box[0]:box[2], :] for box in box_list]

    for i, image in enumerate(image_list):
        image = np.expand_dims(image, 0)
        image = image[:, :, :, ::-1]
        if i == 0:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # input_image = tf.constant(image, dtype=tf.float32)
        input_image = images
        # print(input_image)
        output = build_server_graph(input_image, args.conv_hidden_num, args.input_scale_size)
        output = denorm_img(output)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        # print(result)

    out_image = np.zeros((block_height * block_size, block_width * block_size, 3), dtype=np.uint8)
    for i, box in enumerate(box_list):
        out_image[box[1]:box[3], box[0]:box[2], :] = result[i, :, :, ::-1]

    if need_resize:
        small_img = cv2.resize(out_image, (width, height), interpolation=cv2.INTER_NEAREST)  # (col width, row height)
    else:
        small_img = out_image

    cv2.imwrite(args.output, small_img)


if __name__ == "__main__":
    single_image(parseArguments(sys.argv[1:]))
