import os
from glob import glob
import tensorflow as tf


def get_loader(config, seed=None):
    paths = glob(os.path.join(config.train_path, '*.png'))
    in_size = config.input_scale_size
    # tf_decode = tf.image.decode_jpeg
    tf_decode = tf.image.decode_png
    # tf_decode = tf.image.decode_image

    print(len(paths))

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=True, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)
    image = tf.image.resize_images(image, [in_size, in_size * 2],
                                   method=tf.image.ResizeMethod.AREA)  # [row height 128, col width 256]
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * config.batch_size

    queue = tf.train.shuffle_batch([image], batch_size=config.batch_size, num_threads=64, capacity=capacity,
                                   min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    print('queue:', queue)
    output, inputs = tf.split(queue, [in_size, in_size], 2)
    # inputs = tf.image.crop_to_bounding_box(inputs, *cropbox)
    # inputs = tf.image.resize_images(inputs, [in_size, in_size], method=tf.image.ResizeMethod.AREA)
    # output = tf.image.crop_to_bounding_box(output, *cropbox)
    # output = tf.image.resize_images(output, [in_size, in_size], method=tf.image.ResizeMethod.AREA)
    # print('inputs:', inputs)
    # print('output:', output)
    return tf.to_float(inputs), tf.to_float(output)


if __name__ == "__main__":
    get_loader(r'C:\DATA\_test')
