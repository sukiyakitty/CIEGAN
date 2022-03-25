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
    parser.add_argument('--input', default=r'', type=str,
                        help='The folder of image to be completed.')
    parser.add_argument('--output', default=r'', type=str, help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default=r'', type=str,
                        help='The directory of tensorflow checkpoint.')
    parser.add_argument('--input_scale_size', type=int, default=128)
    parser.add_argument('--conv_hidden_num', type=int, default=128, choices=[128], help='default 128')
    return parser.parse_args(argv)


def images_folder_(args):
    block_size = args.input_scale_size
    if not os.path.isdir(args.input):
        print('!ERROR! The input path does not existed!')
        return False

    img_list = os.listdir(args.input)
    for i, img in enumerate(img_list):
        image = any_to_image(os.path.join(args.input, img))
        height = image.shape[0]
        width = image.shape[1]

        if height < block_size or width < block_size:
            image = cv2.resize(image, (block_size, block_size), interpolation=cv2.INTER_NEAREST)

        image = np.expand_dims(image, 0)
        image = image[:, :, :, ::-1]
        if i == 0:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

    if images.shape[0] > 512:
        images_split = [images[i:i + 512, :, :, :] for i in range(0, images.shape[0], 512)]
        img_list_split = [img_list[i:i + 512] for i in range(0, len(img_list), 512)]

        for k in range(len(images_split)):
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                # input_image = tf.constant(image, dtype=tf.float32)
                input_image = images_split[k]
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

            for i, img in enumerate(img_list_split[k]):
                cv2.imwrite(os.path.join(args.output, img), result[i, :, :, ::-1])
    else:
        pass


def images_folder(args):
    block_size = args.input_scale_size
    if not os.path.isdir(args.input):
        print('!ERROR! The input path does not existed!')
        return False

    img_list = os.listdir(args.input)
    for i, img in enumerate(img_list):
        image = any_to_image(os.path.join(args.input, img))
        height = image.shape[0]
        width = image.shape[1]

        if height < block_size or width < block_size:
            image = cv2.resize(image, (block_size, block_size), interpolation=cv2.INTER_NEAREST)

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

    for i, img in enumerate(img_list):
        cv2.imwrite(os.path.join(args.output, img), result[i, :, :, ::-1])


if __name__ == "__main__":
    images_folder(parseArguments(sys.argv[1:]))
