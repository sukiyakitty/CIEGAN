import os
import tensorflow as tf
from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config):
    # dataset_path = r'C:\DATA\XXXXX'
    prepare_dirs_and_logger(config)

    tf.set_random_seed(config.random_seed)

    data, output = get_loader(config)
    with tf.device('/gpu:0'):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=sess_config) as sess:
            trainer = Trainer(config, data, output, sess)

            if config.is_train:
                save_config(config)
                trainer.train()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
