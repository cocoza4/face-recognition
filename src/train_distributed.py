import os
import math
import time
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial
from pathlib import Path

import utils
import losses
import models
from evaluate import lfw, cfp, eval_utils
from generators import TFRecordDataGenerator

logging.basicConfig(level=logging.INFO)


@tf.function
def predict_embedding(model, images):
    x = model(images, training=False)
    return tf.nn.l2_normalize(x, axis=-1)

@tf.function
def train_step(model, inputs, labels, emb_weights, optimizer, loss_fn, global_batch_size):
    with tf.GradientTape(persistent=False) as tape:
        embeddings = model(inputs, training=True)
        per_example_loss = loss_fn(embeddings, emb_weights, labels)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    trainable_vars = model.trainable_variables + [emb_weights]
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss

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
    _, _, c = image.shape
    if c > 3 or c == 1:
        image = utils.to_rgb(image.numpy())
    return _preprocess(image, training=training)

def load_ckpt(ckpt_dir, max_to_keep, backbone, global_step, model, optimizer, emb_weights):
    ckpt = tf.train.Checkpoint(global_step=global_step, model=model, optimizer=optimizer, emb_weights=emb_weights)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=max_to_keep, checkpoint_name=backbone)
 
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    return ckpt_manager

def evaluate(summary, global_step, eval_fn, predict_fn, tag, results_file):
    t1 = time.time()
    accuracy, val, far, frr = eval_fn(predict_fn=predict_fn)
    time_elapsed = time.time() - t1
    summary.scalar('%s/accuracy' % tag, accuracy, step=global_step)
    summary.scalar('%s/val_rate' % tag, val, step=global_step)
    summary.scalar('%s/far' % tag, far, step=global_step)
    summary.scalar('%s/frr' % tag, frr, step=global_step)
    summary.scalar('%s/time_elapsed' % tag, time_elapsed, step=global_step)
    eval_utils.save_result(results_file, accuracy, val, far, frr)


def get_optimizer(name, lr, **kwargs):
    if name.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(lr)
    elif name.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=kwargs['mom'], nesterov=True)
    return opt

def train(cfg, model, emb_weights, optimizer, learning_rate, global_step, train_ds, strategy, lfw_evaluator, cfp_evaluator):

    def predict_fn(inputs):
        preprocessed = np.array([preprocess(path, training=False).numpy() for path in inputs])
        return predict_embedding(model, preprocessed)

    ckpt_manager = load_ckpt(cfg['ckpt_dir'], cfg['max_to_keep'], cfg['backbone'], global_step, model, optimizer, emb_weights)
    
    current_time = datetime.now().strftime("%Y-%m-%d")
    logdir = os.path.join(cfg['logdir'], current_time)
    summary = tf.compat.v2.summary
    writer = summary.create_file_writer(logdir)

    loss_fn = partial(losses.arcface_loss, n_classes=cfg['n_classes'], m1=cfg['m1'], m2=cfg['m2'], m3=cfg['m3'], s=cfg['s'], 
                        reduction=tf.keras.losses.Reduction.NONE)

    log_template = 'Epoch: %d[%d/%d]\tStep %d\tTime %.3f\tLoss %2.3f\tlr %.5f'

    lfw_results_file = cfg['ckpt_dir'] + '/lfw_results.csv'
    cfp_ff_results_file = cfg['ckpt_dir'] + '/cfp_ff_results.csv'
    cfp_fp_results_file = cfg['ckpt_dir'] + '/cfp_fp_results.csv'

    global_batch_size = cfg['per_replica_batch_size'] * strategy.num_replicas_in_sync
    train_iter = iter(train_ds)
    with writer.as_default():
        for epoch in range(cfg['initial_epoch'], cfg['epochs']):
            
            for step in range(cfg['steps_per_epoch']):
                inputs, labels = next(train_iter)
                t1 = time.time()
                per_replica_loss = strategy.experimental_run_v2(train_step, args=(model, inputs, labels, emb_weights, 
                                                                                    optimizer, loss_fn, global_batch_size,))
                loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                elapsed = time.time() - t1

                current_lr = learning_rate().numpy()
                print(log_template % (epoch+1, step+1, cfg['steps_per_epoch'], global_step.numpy(), elapsed, loss.numpy(), current_lr))
                                        
                summary.scalar('train/loss', loss, step=global_step)
                global_step.assign_add(1)

            if epoch % cfg['eval_every'] == 0 or epoch == epochs-1:
                # lfw
                evaluate(summary, global_step, lfw_evaluator.evaluate, predict_fn, 'lfw', lfw_results_file)
                
                # cfp front face
                evaluate(summary, global_step, cfp_evaluator.evaluate_ff, predict_fn, 'cfp_ff', cfp_ff_results_file)

                # cfp front profile
                evaluate(summary, global_step, cfp_evaluator.evaluate_fp, predict_fn, 'cfp_fp', cfp_fp_results_file)

            writer.flush()
            
            try:

                # CheckpointManager throws the following error when saving model checkpoint

                # Failed copying input tensor from /job:localhost/replica:0/task:0/device:GPU:0 to 
                # /job:localhost/replica:0/task:0/device:CPU:0 in order to run Identity: Dst tensor is not initialized. 
                # [Op:Identity add try-catch if error persists
                
                # This is due to insufficient RAM (not GPU Ram), thus adding more ram fixes the issue After some research.
                # Wrap in try-catch statement to avoid failure.
                save_path = ckpt_manager.save()
                logging.info('Saving checkpoint for epoch {} at {}'.format(epoch+1, save_path))
            except Exception as ex:
                print(ex)


def main():
    cfg = utils.load_config_file(args.config_path)
    utils.save_configs(cfg['ckpt_dir'] + '/configs.txt', cfg)

    cfp_ff_dir = Path(cfg['cfp_ff_dir'])
    cfp_fp_dir = Path(cfg['cfp_fp_dir'])
    ff_fold_files = list(cfp_ff_dir.glob('ff*.csv'))
    fp_fold_files = list(cfp_fp_dir.glob('fp*.csv'))
    ff_mapping_file = cfp_ff_dir / 'pair_list_ff.csv'
    fp_mapping_file = cfp_fp_dir / 'pair_list_fp.csv'

    cfp_evaluator = cfp.CFPEvaluator(cfp_ff_dir, cfp_fp_dir, ff_fold_files, fp_fold_files, ff_mapping_file, 
                                    fp_mapping_file, cfg['embedding_size'], cfg['test_batch_size'])

    lfw_evaluator = lfw.LFWEvaluator(cfg['lfw_dir'], cfg['lfw_pairs'], 
                                    batch_size=cfg['test_batch_size'], embedding_size=cfg['embedding_size'], 
                                    n_folds=10)

    strategy = tf.distribute.MirroredStrategy()

    # TF Records
    batch_size = cfg['per_replica_batch_size'] * strategy.num_replicas_in_sync
    train_gen = TFRecordDataGenerator(cfg['train_dir'], batch_size=batch_size)
    train_ds = train_gen.generate(example_parser=parse_example, preprocess_fn=preprocess_tf_example)

    with strategy.scope():
        
        # model and optimizer must be created under `strategy.scope`.
        global_step = tf.Variable(0, name="global_step", dtype=tf.int64, trainable=False)
        learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries=cfg['lr_steps'], values=cfg['lr_values'], name='lr_schedule')
        optimizer = get_optimizer(cfg['optimizer'], learning_rate, mom=cfg['mom'])

        model = models.create_model(cfg['backbone'], cfg['embedding_size'])
        initializer = tf.initializers.VarianceScaling()
        emb_weights = tf.Variable(initializer(shape=[cfg['n_classes'], cfg['embedding_size']]), 
                                    name='embedding_weights', dtype=tf.float32)
        
        train_dist_ds = strategy.experimental_distribute_dataset(train_ds)

        train(cfg, model, emb_weights, optimizer, learning_rate, global_step, train_dist_ds, strategy, lfw_evaluator, cfp_evaluator)

    logging.info('Trained successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, help='Path to config file', required=True)

    args = parser.parse_args()
    main()
