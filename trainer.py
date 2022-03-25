import os
import tensorflow as tf
from models import GeneratorCNN, build_vgg19, build_style_loss, total_variation_loss, DiscriminatorCNNg, denorm_img
from utils import save_image, norm_img, get_time


class Trainer(object):
    def __init__(self, config, data, output, sess):
        self.sess = sess
        self.config = config
        self.data = data
        self.output = output
        self.data_dir = config.data_dir
        self.dataset = config.dataset
        self.weight_decay_rate = 0.00001
        self.lambda_r = config.lambda_r
        self.lambda_p = config.lambda_p
        self.lambda_a = config.lambda_a
        self.lambda_c = config.lambda_c
        self.lambda_s = config.lambda_s
        self.lambda_t = config.lambda_t

        self.step_1 = config.step_1
        self.step_2 = config.step_2
        self.step_3 = config.step_3

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.max_to_keep = 1

        self.using_loss_G_r = config.using_loss_G_r
        self.using_loss_G_p = config.using_loss_G_p
        self.using_loss_G_a = config.using_loss_G_a

        self.step = tf.Variable(0, name='step', trainable=False)
        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
                                     name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
                                     name='d_lr_update')

        self.gamma = config.gamma

        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        coord, threads = self.queue_context()
        self.build_model()

    def queue_context(self):
        # thread coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        return coord, threads

    def build_model(self):
        x = self.data
        output = self.output

        self.inputs = norm_img(x)
        self.output = norm_img(output)

        y_real = tf.ones(self.batch_size)
        y_fake = tf.zeros(self.batch_size)

        G, self.G_var = GeneratorCNN(self.inputs, self.conv_hidden_num, reuse=False)

        vgg_real = build_vgg19(self.output)
        vgg_fake = build_vgg19(G)
        size = tf.size(vgg_real['conv1_2'])
        p1 = tf.nn.l2_loss(vgg_real['conv1_2'] - vgg_fake['conv1_2']) * 2 / tf.to_float(size)
        size = tf.size(vgg_real['conv2_2'])
        p2 = tf.nn.l2_loss(vgg_real['conv2_2'] - vgg_fake['conv2_2']) * 2 / tf.to_float(size)
        size = tf.size(vgg_real['conv3_2'])
        p3 = tf.nn.l2_loss(vgg_real['conv3_2'] - vgg_fake['conv3_2']) * 2 / tf.to_float(size)
        self.content_loss = p1 + p2 + p3
        style1 = build_style_loss(vgg_real['conv1_2'], vgg_fake['conv1_2'])
        style2 = build_style_loss(vgg_real['conv2_2'], vgg_fake['conv2_2'])
        style3 = build_style_loss(vgg_real['conv3_2'], vgg_fake['conv3_2'])
        self.style_loss = style1 + style2 + style3
        self.tv = total_variation_loss(G)

        disc_real, self.D_var = DiscriminatorCNNg(self.output, self.conv_hidden_num)
        disc_fake, _ = DiscriminatorCNNg(G, self.conv_hidden_num)

        self.G = denorm_img(G)
        self.inputdenorm = denorm_img(self.inputs)
        self.ground = denorm_img(self.output)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer

        G_optimizer = optimizer(self.g_lr, beta1=0.5)
        D_optimizer = optimizer(self.d_lr, beta1=0.5)

        Loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=y_real))
        Loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_fake))
        self.Loss_D = Loss_D_real + Loss_D_fake

        self.Loss_G_reconstruction = tf.reduce_mean(tf.square(self.output - G))
        self.Loss_G_perceptual = self.lambda_c * self.content_loss + self.lambda_s * self.style_loss + self.lambda_t * self.tv
        self.Loss_G_adversarial = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_real))

        self.Loss_G_r_p = self.lambda_r * self.Loss_G_reconstruction + self.lambda_p * self.Loss_G_perceptual
        self.Loss_G = self.lambda_r * self.Loss_G_reconstruction + self.lambda_p * self.Loss_G_perceptual + self.lambda_a * self.Loss_G_adversarial

        self.Optim_G_step1 = G_optimizer.minimize(self.Loss_G_r_p, global_step=self.step, var_list=self.G_var)
        self.Optim_G_step2 = G_optimizer.minimize(self.Loss_G, global_step=self.step, var_list=self.G_var)
        self.Optim_D = D_optimizer.minimize(self.Loss_D, var_list=self.D_var)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=(self.max_to_keep))

        if self.is_train:
            tf.summary.image("G", self.G),
            tf.summary.scalar("loss/Loss_G_reconstruction", self.Loss_G_reconstruction),
            tf.summary.scalar("loss/Loss_G_content", self.content_loss),
            tf.summary.scalar("loss/Loss_G_cstyle", self.style_loss),
            tf.summary.scalar("loss/Loss_G_tv", self.tv),
            tf.summary.scalar("loss/Loss_G_perceptual", self.Loss_G_perceptual),
            tf.summary.scalar("loss/Loss_G_adversarial", self.Loss_G_adversarial),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("loss/Loss_G", self.Loss_G),
            tf.summary.scalar("loss/Loss_D", self.Loss_D),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            self.merged = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

    def train(self):
        save_path_name = os.path.join('checkpoint_' + get_time(), self.dataset)
        for step in range(self.start_step, self.max_step):
            if step < self.step_3:
                if step < self.step_1:
                    self.sess.run(self.Optim_G_step1)
                else:
                    self.sess.run(self.Optim_D)
                    if step > self.step_2:
                        self.sess.run(self.Optim_G_step2)

                if step % (self.log_step) == 0:
                    Loss_G_r = self.sess.run(self.Loss_G_reconstruction)
                    Loss_G_c = self.sess.run(self.content_loss)
                    Loss_G_s = self.sess.run(self.style_loss)
                    Loss_G_t = self.sess.run(self.tv)
                    Loss_G_p = self.sess.run(self.Loss_G_perceptual)
                    Loss_G_a = self.sess.run(self.Loss_G_adversarial)
                    Loss_D = self.sess.run(self.Loss_D)
                    Loss_G = self.sess.run(self.Loss_G)
                    val = self.sess.run(self.merged)
                    self.summary_writer.add_summary(val, global_step=step)
                    print(
                        "[{}/{}] Loss_G_reconstruction: {:.6f} Loss_G_content: {:.6f} Loss_G_style: {:.6f} Loss_G_tv: {:.6f} Loss_G_perceptual: {:.6f} Loss_G_adversarial: {:.6f} Loss_G: {:.6f} Loss_D: {:.6f}"
                            .format(step, self.max_step, Loss_G_r, Loss_G_c, Loss_G_s, Loss_G_t, Loss_G_p, Loss_G_a,
                                    Loss_G, Loss_D))

                if step % (self.save_step) == 0:
                    self.saver.save(self.sess, save_path_name, global_step=step)
                    x1, x2, x3 = self.sess.run([self.G, self.inputdenorm, self.ground])
                    path = os.path.join(self.data_dir, '{}_generated.png'.format(step))
                    save_image(x1, path)
                    path = os.path.join(self.data_dir, '{}_input.png'.format(step))
                    save_image(x2, path)
                    path = os.path.join(self.data_dir, '{}_ground_truth.png'.format(step))
                    save_image(x3, path)
                    print("[*] Samples saved: {}".format(path))
