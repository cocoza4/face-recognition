import os
import csv
import math
import time
import yaml
import argparse
import logging
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from tensorflow.keras.applications import densenet
from datetime import datetime

import lfw
import losses
from models import ArcFaceModel
from generators import DataGenerator, TFRecordDataGenerator


@tf.function
def train_step(model, inputs, labels, emb_weights, optimizer, n_classes, m1, m2, m3, s):
    with tf.GradientTape(persistent=False) as tape:
        embeddings = model(inputs, training=True)
        loss = losses.arcface_loss(embeddings, emb_weights, labels, n_classes, m1, m2, m3, s)
    
    trainable_vars = model.trainable_variables + [emb_weights]
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss

@tf.function
def test_step(model, images, labels, n_classes, m1, m2, m3, s):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    embeddings = model(inputs, training=True) # training mode as loss is calculated
    loss = losses.arcface_loss(embeddings, emb_weights, labels, n_classes, m1, m2, m3, s)
    return loss

@tf.function
def predict_embedding(model, images):
    embeddings = model(images, training=False)
    return embeddings

def parse_example(proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/label': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64)
    }
    tf_example = tf.io.parse_single_example(proto, feature_description)
    return tf_example

def _preprocess(image, training):
    if training:
        image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image -= 127.5
    image *= 0.0078125
    return image

def preprocess_tf_example(example, training=True):
    width = example['image/width']
    height = example['image/height']
    label = tf.cast(example['image/label'], tf.int32)
    image = tf.io.decode_image(example['image/encoded'])
    return _preprocess(image, training=training), label
    

def preprocess(path, training=True):
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw)
    return _preprocess(image, training=training)


def evaluate(model, lfw_paths, actual_issame, batch_size, embedding_size, n_folds):
    n_images = len(actual_issame) * 2
    assert len(lfw_paths) == n_images

    embs_array = np.zeros((n_images, embedding_size))
    it = tqdm(range(0, n_images, batch_size), 'evaluate on LFW')
    for start in it:
        end = start + batch_size
        preprocessed = np.array([preprocess(path, training=False).numpy() for path in lfw_paths[start:end]])
        embs_array[start:end] = predict_embedding(model, preprocessed)
        
    _, _, accuracy, val, val_std, far, frr = lfw.evaluate(embs_array, actual_issame, n_folds=n_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, FRR=%2.5f' % (val, val_std, far, frr))

    return np.mean(accuracy), val, far, frr

def save_configs(config):
    with open(config['ckpt_dir'] + '/configs.txt', 'w') as f:
        f.write('Parameters\n')
        for arg, value in config.items():
            f.write('{}: {}\n'.format(arg, value))

def save_lfw_result(path, accuracy, val, far, frr):
    columns = ['accuracy', 'val', 'far', 'frr']
    lfw_file = path + '/lfw_results.csv'
    exists = os.path.exists(lfw_file)
    with open(lfw_file, mode='a+') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({'accuracy': accuracy, 'val': val, 'far': far, 'frr':frr})

def load_config_file(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        return config


def main():
    config = load_config_file(args.config_path)
    save_configs(config)

    initial_epoch = config['initial_epoch']
    epochs = config['epochs']

    logging.info('creating model')
    backbone = tf.keras.applications.DenseNet121(weights=None, include_top=False, pooling='avg')
    # backbone = tf.keras.applications.ResNet101(weights=None, include_top=False, pooling='avg')
    # backbone = tf.keras.applications.ResNet101V2(weights=None, include_top=False, pooling='avg')
    
    model = ArcFaceModel(backbone, config['embedding_size'])

    initializer = tf.initializers.VarianceScaling()
    emb_weights = tf.Variable(initializer(shape=[config['n_classes'], config['embedding_size']]), 
                                name='embedding_weights', dtype=tf.float32)

    global_step = tf.Variable(0, name="global_step", dtype=tf.int64, trainable=False)

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,
    #             decay_steps=args.lr_decay_steps,
    #             decay_rate=args.lr_decay_rate,
    #             staircase=True)
    # optimizer = tf.keras.optimizers.Adam(lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries=config['lr_steps'], values=config['lr_values'], name='lr_schedule')
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    current_time = datetime.now().strftime("%Y-%m-%d")
    logdir = os.path.join(config['logdir'], current_time)

    summary = tf.compat.v2.summary
    writer = summary.create_file_writer(logdir)

    ckpt = tf.train.Checkpoint(global_step=global_step, backbone=model.backbone, model=model, optimizer=optimizer, emb_weights=emb_weights)
    ckpt_manager = tf.train.CheckpointManager(ckpt, config['ckpt_dir'], max_to_keep=config['max_to_keep'], checkpoint_name=config['backbone'])

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    logging.info('reading lfw data')
    pairs = lfw.read_pairs(config['lfw_pairs'])
    lfw_paths, actual_issame = lfw.get_paths(config['lfw_dir'], pairs)

    # train_gen = DataGenerator(args.data_dir, batch_size=args.batch_size)
    # train_gen_it = train_gen.generate(preprocess_fn=preprocess)
    # steps_per_epoch = train_gen.steps_per_epoch()

    # TF Records
    train_gen = TFRecordDataGenerator(config['data_dir'], batch_size=config['batch_size'])
    train_gen_it = train_gen.generate(example_parser=parse_example, preprocess_fn=preprocess_tf_example)
    # steps_per_epoch = train_gen.steps_per_epoch()

    with writer.as_default():
        for epoch in range(initial_epoch, epochs):
            for step, (inputs, targets) in enumerate(train_gen_it):
                t1 = time.time()
                loss = train_step(model, inputs, targets, emb_weights, optimizer, 
                                config['n_classes'], config['m1'], config['m2'], config['m3'], config['s'])
                elapsed = time.time() - t1
                # current_lr = lr_schedule(global_step)

                # current_lr = optimizer.lr.numpy()
                # summary.scalar('learning_rate', current_lr, step=global_step)
                # # print('Epoch: %d[%d/%d]\tStep %d\tTime %.3f\tLoss %2.3f\tlr %.5f' % 
                # #     (epoch+1, step+1, steps_per_epoch, global_step, elapsed, loss, current_lr))

                current_lr = learning_rate().numpy()
                summary.scalar('learning_rate', current_lr, step=global_step)
                print('Epoch: %d[%d/%d]\tStep %d\tTime %.3f\tLoss %2.3f\tlr %.5f' % 
                    (epoch+1, step+1, config['steps_per_epoch'], global_step, elapsed, loss, current_lr))

                summary.scalar('train/loss', loss, step=global_step)
                global_step.assign_add(1)

            if epoch % config['eval_every'] == 0 or epoch == epochs-1:
                t1 = time.time()
                accuracy, val, far, frr = evaluate(model, lfw_paths, actual_issame, config['batch_size'], config['embedding_size'], config['lfw_n_folds'])
                time_elapsed = time.time() - t1
                summary.scalar('lfw/accuracy', accuracy, step=global_step)
                summary.scalar('lfw/val_rate', val, step=global_step)
                summary.scalar('lfw/far', far, step=global_step)
                summary.scalar('lfw/frr', frr, step=global_step)
                summary.scalar('lfw/time_elapsed', time_elapsed, step=global_step)
                save_lfw_result(config['ckpt_dir'], accuracy, val, far, frr)

            writer.flush()
            save_path = ckpt_manager.save()
            logging.info('Saving checkpoint for epoch {} at {}'.format(epoch+1, save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, help='Path to config file', required=True)

    # parser.add_argument('--gpu_memory_fraction', type=float, 
    #                     help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    # parser.add_argument("--ckpt_dir", help="Base model checkpoints directory.", required=True)
    # parser.add_argument("--max_to_keep", help="Base model checkpoints directory.", type=int, default=10)
    # parser.add_argument("--n_classes", help="Number of classes(faces).", type=int, required=True)
    # parser.add_argument("--batch_size", help="Batch size.", type=int, default=256)
    # parser.add_argument("--backbone", help="Backbone model.", default='densenet121')
    # parser.add_argument("--img_width", help="Image Width.", type=int, default=112)
    # parser.add_argument("--img_height", help="Image Height.", type=int, default=112)
    # parser.add_argument("--data_dir", help="Data directory.", 
    #                     default="/mnt/disks/data/datasets/vggface2/train_mtcnnpy_112_margin32")
    # parser.add_argument("--embedding_size", help="Face embedding size.", type=int, default=512)
    # parser.add_argument("--m1", help="m1.", type=float, default=1.0)
    # parser.add_argument("--m2", help="m2.", type=float, default=0.3)
    # parser.add_argument("--m3", help="m3.", type=float, default=0.2)
    # parser.add_argument("--s", help="s.", type=float, default=64.)
    # parser.add_argument("--initial_epoch", help="Initial epoch.", type=int, default=0)
    # parser.add_argument("--epochs", help="Number of epochs.", type=int, default=100)
    # parser.add_argument('--learning_rate', help='learning rate.', type=float, default=0.001)
    # parser.add_argument("--lr_decay_steps", help="Number of steps to decay the learning rate to another step.", type=int, default=5000)
    # parser.add_argument("--lr_decay_rate", help="Learning rate decay rate.", type=float, default=0.96)

    # parser.add_argument("--lfw_pairs", help="The file containing the pairs to use for validation.", 
    #                     default="/mnt/disks/data/datasets/lfw/raw_mtcnnpy_160/pairs.txt")
    # parser.add_argument("--lfw_dir", help="Path to the data directory containing aligned face patches.", 
    #                     default="/mnt/disks/data/datasets/lfw/raw_mtcnnpy_160")
    # parser.add_argument("--eval_every", help="Evaluate on LFW data every n epochs.", type=int, default=1)

    # parser.add_argument("--logdir", help="Log directory.", required=True)
    # parser.add_argument("--lfw_n_folds", help="Number of folds to use for cross validation. Mainly used for testing.", 
    #                     type=int, default=10)

    args = parser.parse_args()
    main()
