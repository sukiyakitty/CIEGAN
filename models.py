import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

slim = tf.contrib.slim
IMAGENET_VGG19 = r'C:\DATA\MODEL\imagenet-vgg-verydeep-19.mat'


def build_server_graph(inputs, conv_hidden_num=128, input_scale_size=128, reuse=False):
    inputs = tf.image.resize_images(inputs, [input_scale_size, input_scale_size], method=tf.image.ResizeMethod.AREA)
    inputs = tf.to_float(inputs)
    inputs = inputs / 127.5 - 1
    x, var = GeneratorCNN(inputs, conv_hidden_num, reuse)
    batch_predict = x
    return batch_predict


def denorm_img(norm):
    return tf.clip_by_value((norm + 1) * 127.5, 0, 255)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def GeneratorCNN(maskx, hidden_num, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        x = slim.conv2d(maskx, hidden_num, 5, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 128, 3, 2, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 256, 3, 2, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 256, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 256, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = tf.contrib.layers.conv2d(x, 256, 3, 1, activation_fn=tf.identity, rate=2)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = tf.contrib.layers.conv2d(x, 256, 3, 1, activation_fn=tf.identity, rate=4)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = tf.contrib.layers.conv2d(x, 256, 3, 1, activation_fn=tf.identity, rate=8)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = tf.contrib.layers.conv2d(x, 256, 3, 1, activation_fn=tf.identity, rate=16)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 256, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 256, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)

        x = slim.conv2d_transpose(x, 128, 4, 2, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d_transpose(x, 64, 4, 2, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        x = slim.conv2d(x, 32, 3, 1, activation_fn=tf.identity)
        x = slim.batch_norm(x, activation_fn=tf.identity)
        x = leaky_relu(x)
        # x = attention(x, 64)
        out = slim.conv2d(x, 3, 3, 1, activation_fn=tf.nn.tanh)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables


def DiscriminatorCNNl(x, hidden_num):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Dl')]) > 0
    with tf.variable_scope("Dl", reuse=reuse) as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 128, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 256, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 512, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 512, 5, 2, activation_fn=tf.nn.relu)
        x = tf.reshape(x, [-1, np.prod([3, 3, 512])])
        x = slim.fully_connected(x, 1024, activation_fn=tf.nn.relu)
        disc = slim.fully_connected(x, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)
    variables = tf.contrib.framework.get_variables(vs)
    return disc, variables


def DiscriminatorCNNg(x, hidden_num):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Dg')]) > 0
    with tf.variable_scope("Dg", reuse=reuse) as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 128, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 256, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 512, 5, 2, activation_fn=tf.nn.relu)
        x = slim.conv2d(x, 512, 5, 2, activation_fn=tf.nn.relu)
        # x = attention(x, 512)
        x = tf.layers.flatten(x)
        # x = tf.reshape(x, [-1, np.prod([4, 4, 512])])
        x = slim.fully_connected(x, 1024, activation_fn=tf.nn.relu)
        disc = slim.fully_connected(x, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)
    variables = tf.contrib.framework.get_variables(vs)
    return disc, variables


def DiscriminatorCNNall(xg, xl, hidden_num):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('D')]) > 0
    with tf.variable_scope("D", reuse=reuse) as vs:
        # discl
        xl = slim.conv2d(xl, hidden_num, 5, 2, activation_fn=tf.identity)
        xl = slim.batch_norm(xl, activation_fn=tf.identity)
        xl = leaky_relu(xl)
        xl = slim.conv2d(xl, 128, 5, 2, activation_fn=tf.identity)
        xl = slim.batch_norm(xl, activation_fn=tf.identity)
        xl = leaky_relu(xl)
        xl = slim.conv2d(xl, 256, 5, 2, activation_fn=tf.identity)
        xl = slim.batch_norm(xl, activation_fn=tf.identity)
        xl = leaky_relu(xl)
        xl = slim.conv2d(xl, 512, 5, 2, activation_fn=tf.identity)
        xl = slim.batch_norm(xl, activation_fn=tf.identity)
        xl = leaky_relu(xl)
        xl = tf.reshape(xl, [-1, np.prod([4, 4, 512])])
        xl = slim.fully_connected(xl, 1024, activation_fn=tf.identity)
        xl = slim.batch_norm(xl, activation_fn=tf.identity)
        xl = leaky_relu(xl)

        # discg
        xg = slim.conv2d(xg, hidden_num, 5, 2, activation_fn=tf.identity)
        xg = slim.batch_norm(xg, activation_fn=tf.identity)
        xg = leaky_relu(xg)
        xg = slim.conv2d(xg, 128, 5, 2, activation_fn=tf.identity)
        xg = slim.batch_norm(xg, activation_fn=tf.identity)
        xg = leaky_relu(xg)
        xg = slim.conv2d(xg, 256, 5, 2, activation_fn=tf.identity)
        xg = slim.batch_norm(xg, activation_fn=tf.identity)
        xg = leaky_relu(xg)
        xg = slim.conv2d(xg, 512, 5, 2, activation_fn=tf.identity)
        xg = slim.batch_norm(xg, activation_fn=tf.identity)
        xg = leaky_relu(xg)
        xg = slim.conv2d(xg, 512, 5, 2, activation_fn=tf.identity)
        xg = slim.batch_norm(xg, activation_fn=tf.identity)
        xg = leaky_relu(xg)
        xg = tf.reshape(xg, [-1, np.prod([4, 4, 512])])
        xg = slim.fully_connected(xg, 1024, activation_fn=tf.identity)
        xg = slim.batch_norm(xg, activation_fn=tf.identity)
        xg = leaky_relu(xg)

        x = tf.concat([xg, xl], 1)
        print(x)
        disc = slim.fully_connected(x, num_outputs=1, activation_fn=None)
        disc = tf.squeeze(disc, -1)
    variables = tf.contrib.framework.get_variables(vs)
    return disc, variables


def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i, ):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][0][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def build_vgg19(input):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('vgg19')]) > 0
    with tf.variable_scope("vgg19", reuse=tf.AUTO_REUSE) as vs:
        net = {}
        vgg_rawnet = scipy.io.loadmat(IMAGENET_VGG19)
        vgg_layers = vgg_rawnet['layers'][0]
        net['input'] = input
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
        net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
        net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
        net['pool5'] = build_net('pool', net['conv5_4'])
    variables = tf.contrib.framework.get_variables(vs)
    return net


def gram(x):
    shape = tf.shape(x)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(x, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def build_style_loss(real, fake):
    x = gram(real)
    G = gram(fake)
    size = tf.size(x)
    loss = tf.nn.l2_loss(x - G) * 2 / tf.to_float(size)
    return loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0],
                                                                                     [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0],
                                                                                    [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
