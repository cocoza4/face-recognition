import os
import math
import time
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


@tf.function
def train_step(model, inputs, labels, optimizer):
    with tf.GradientTape(persistent=False) as tape:
        prelogits, norm_dense = model(inputs, training=True)
        loss = losses.arcface_loss(prelogits, norm_dense, labels, args.m1, args.m2, args.m3, args.s)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

@tf.function
def test_step(model, images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    prelogits, norm_dense = model(images, training=True) # training mode as loss is calculated
    loss = losses.arcface_loss(prelogits, norm_dense, labels, args.m1, args.m2, args.m3, args.s)

    return loss

@tf.function
def predict_embedding(model, images):
    embeddings = model(images, training=False)
    return embeddings


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def imagenet_preprocess(x):
    # image /= 255.
    # image[..., 0] -= mean[0]
    # image[..., 1] -= mean[1]
    # image[..., 2] -= mean[2]
    # image[..., 0] /= std[0]
    # image[..., 1] /= std[1]
    # image[..., 2] /= std[2]
    x /= 255.
    x = tf.stack([x[..., 0] - mean[0], x[..., 1] - mean[1], x[..., 2] - mean[2]], axis=-1)
    x = tf.stack([x[..., 0] / std[0], x[..., 1] / std[1], x[..., 2] / std[2]], axis=-1)
    return x


def preprocess(path, training=True):
    raw = tf.io.read_file(path)
    image = tf.image.decode_png(raw)
    if training:
        image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image = imagenet_preprocess(image)
    image = tf.image.resize(image, (args.img_width, args.img_height))

    # image = tf.image.resize(image, (224, 224))
    # image = tf.image.random_crop(image, size=[112, 112, 3])
    return image


def get_data(path):
    ids = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    ids.sort()
    cat_num = len(ids)

    id_dict = dict(zip(ids, list(range(cat_num))))
    paths = []
    labels = []
    for i in ids:
        cur_dir = os.path.join(path, i)
        fns = os.listdir(cur_dir)
        paths.append([os.path.join(cur_dir, fn) for fn in fns])
        labels.append([id_dict[i]] * len(fns))

    return paths, labels

def generate(path, preprocess_fn=preprocess, batch_size=128):
    paths, labels = get_data(path)
    n_classes = len(paths)
    paths = [path for cls in paths for path in cls]
    labels = [label for cls in labels for label in cls]
    
    assert (len(paths) == len(labels))
    def p(inputs, labels): return preprocess(inputs), labels
    
    ds = (tf.data.Dataset.from_tensor_slices((paths, labels))
          .cache()
          .shuffle(20000)
          .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
          .map(p, num_parallel_calls=tf.data.experimental.AUTOTUNE)
          .batch(batch_size))
    
    return ds

def evaluate(model, lfw_paths, actual_issame, batch_size, n_folds):
    n_images = len(actual_issame) * 2
    assert len(lfw_paths) == n_images

    embs_array = np.zeros((n_images, args.embedding_size))
    it = tqdm(range(0, n_images, batch_size), 'Predict embeddings')
    for start in it:
        end = start + batch_size
        preprocessed = np.array([preprocess(path, training=False).numpy() for path in lfw_paths[start:end]])
        embs_array[start:end] = predict_embedding(model, preprocessed)
        
    _, _, accuracy, val, val_std, far, frr = lfw.evaluate(embs_array, actual_issame, n_folds=n_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, FRR=%2.5f' % (val, val_std, far, frr))

    return np.mean(accuracy), val, far, frr


def main():

    logging.info('creating model')
    backbone = tf.keras.Sequential([
        tf.keras.applications.DenseNet121(include_top=False, pooling='avg'),
        tf.keras.layers.Dense(args.embedding_size)
    ])
    model = ArcFaceModel(backbone, args.n_classes)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,
                decay_steps=args.lr_decay_steps,
                decay_rate=args.lr_decay_rate,
                staircase=True)

    optimizer = tf.keras.optimizers.Adam()            

    current_time = datetime.now().strftime("%Y-%m-%d")
    logdir = os.path.join(args.logdir, current_time)

    summary = tf.compat.v2.summary
    writer = summary.create_file_writer(logdir)

    ckpt = tf.train.Checkpoint(backbone=model.backbone, model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=10, checkpoint_name=args.backbone)

    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)

    logging.info('reading lfw data')
    pairs = lfw.read_pairs(args.lfw_pairs)
    lfw_paths, actual_issame = lfw.get_paths(args.lfw_dir, pairs)

    train_gen = generate(args.data_dir, batch_size=args.batch_size)

    with writer.as_default():
        for epoch in range(args.epochs):

            for inputs, targets in train_gen:
                t1 = time.time()
                loss = train_step(model, inputs, targets, optimizer)
                elapsed = time.time() - t1
                # current_lr = lr_schedule(global_step)
                # summary.scalar('learning_rate', current_lr, step=global_step)
                print('Epoch: %d\tStep: %d\tTime %.3f\tLoss %2.3f' % 
                    (epoch+1, global_step, elapsed, loss))

                summary.scalar('train/loss', loss, step=global_step)

                # print('Epoch: %d\tStep: %d\tTime %.3f\tLoss %2.3f\tLearning Rate: %.5f' % 
                #     (epoch+1, global_step, elapsed, loss, current_lr))

                global_step.assign_add(1)

            if epoch % args.eval_every == 0 or epoch == args.epochs-1:
                t1 = time.time()
                accuracy, val, far, frr = evaluate(model, lfw_paths, actual_issame, args.batch_size, args.lfw_n_folds)
                time_elapsed = time.time() - t1
                summary.scalar('lfw/accuracy', accuracy, step=global_step)
                summary.scalar('lfw/val_rate', val, step=global_step)
                summary.scalar('lfw/far', far, step=global_step)
                summary.scalar('lfw/frr', frr, step=global_step)
                summary.scalar('lfw/time_elapsed', time_elapsed, step=global_step)

            writer.flush()
            save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--gpu_memory_fraction', type=float, 
    #                     help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    parser.add_argument("--ckpt_dir", help="Base model checkpoints directory.", required=True)
    parser.add_argument("--n_classes", help="Number of classes(faces).", type=int, required=True)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=256)
    parser.add_argument("--backbone", help="Backbone model.", default='densenet101')
    parser.add_argument("--img_width", help="Image Width.", type=int, default=112)
    parser.add_argument("--img_height", help="Image Height.", type=int, default=112)
    parser.add_argument("--data_dir", help="Data directory.", 
                        default="/mnt/disks/data/datasets/vggface2/train_mtcnnpy_112_margin32")
    parser.add_argument("--embedding_size", help="Face embedding size.", type=int, default=512)
    parser.add_argument("--m1", help="m1.", type=float, default=1.0)
    parser.add_argument("--m2", help="m2.", type=float, default=0.2)
    parser.add_argument("--m3", help="m3.", type=float, default=0.3)
    parser.add_argument("--s", help="s.", type=float, default=64.)
    parser.add_argument("--epochs", help="Number of epochs.", type=int, default=100)
    parser.add_argument('--learning_rate', help='Initial learning rate.', type=float, default=0.01)
    parser.add_argument("--lr_decay_steps", help="Number of steps to decay the learning rate to another step.", type=int, default=100000)
    parser.add_argument("--lr_decay_rate", help="Learning rate decay rate.", type=float, default=0.96)

    parser.add_argument("--lfw_pairs", help="The file containing the pairs to use for validation.", 
                        default="/mnt/disks/data/datasets/lfw/raw_mtcnnpy_160/pairs.txt")
    parser.add_argument("--lfw_dir", help="Path to the data directory containing aligned face patches.", 
                        default="/mnt/disks/data/datasets/lfw/raw_mtcnnpy_160")
    parser.add_argument("--eval_every", help="Evaluate on LFW data every n epochs.", type=int, default=1)

    parser.add_argument("--logdir", help="Log directory.", required=True)
    parser.add_argument("--lfw_n_folds", help="Number of folds to use for cross validation. Mainly used for testing.", 
                        type=int, default=10)

    args = parser.parse_args()
    main()    

    
# python train_arcface.py --n_classes=8631 --batch_size=256 \
#     --ckpt_dir=/home/peeranat_absoroute_io/trained_models/arcface_models \
#     --data_dir=/mnt/disks/data/datasets/vggface2/train_mtcnnpy_112_margin32 \
#     --logdir=/home/peeranat_absoroute_io/trained_models/arcface_models/logs \
#     --lfw_pairs=/mnt/disks/data/datasets/lfw-deepfunneled_mtcnnpy_112_margin32/pairs.txt \
#     --lfw_dir=/mnt/disks/data/datasets/lfw-deepfunneled_mtcnnpy_112_margin32
