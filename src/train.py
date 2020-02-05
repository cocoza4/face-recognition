import os
import time
import random
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tqdm.auto import tqdm

import utils
import lfw
from model import Model


class Trainer:

    def __init__(self, loss, learning_rate, global_step, moving_average_decay):
        
        self.loss = loss
        self.global_step = global_step
        self.learning_rate = learning_rate
        loss_averages_op = self.add_loss_summaries(loss)
    
        with tf.control_dependencies([loss_averages_op]):
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
            # opt = tf.compat.v1.train.AdagradOptimizer(learning_rate)
            grads = self.opt.compute_gradients(loss, tf.compat.v1.global_variables())
            
        apply_gradient_op = self.opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            self.train_op = tf.no_op(name='train')

    def add_loss_summaries(self, total_loss):
        """Add summaries for losses and generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.compat.v1.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summmary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.compat.v1.summary.scalar(l.op.name + '_raw', l)
            tf.compat.v1.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op


def preprocess(inp, image_size=[160, 160]):
    image = inp / 255.
    return tf.image.resize(image, image_size)


def get_people_for_batch(people, idx_batch, people_per_batch):
    start = idx_batch * people_per_batch
    end = min(start + people_per_batch, len(people))
    return people[start:end]


def sample_faces(dataset_root, people, faces_per_person):
    """
    Produces a flat list of file paths to the face images of all people
    in people, each with faces_per_person files.
    """
    images = []
    
    for person in people:
        paths = []
        person_path = os.path.join(dataset_root, person)
        
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            
            # ensure file is valid, and should be more than 1000 bytes
            if (os.path.isfile(img_path)) and (os.path.getsize(img_path) > 1000):  
                paths.append(img_path)
            
        if len(paths) >= faces_per_person:
            samples = random.sample(paths, faces_per_person)
            
            images.extend(samples)
        
    return images


# def read_image(file):
#     return np.asarray(Image.open(file), dtype=np.float32)


def read_image(file):
    img = Image.open(file)
    return np.asarray(img)


def predict_embeddings(model, sess, images, batch_size):
    n_images = len(images)
    batches = int(np.ceil(n_images / batch_size))
    embs_array = np.zeros((n_images, model.emb_size))
    
    it = tqdm(range(batches), 'Predict embeddings')
    for i in it:
        start = i * batch_size
        end = start + batch_size
        images_batch = images[start:end]
        images_batch = [read_image(file) for file in images_batch]
        
        embs = sess.run(model.embeddings, feed_dict={model.input_tensor: images_batch})
        embs_array[start:end] = embs
        
    return embs_array


def get_triplets_images(files, triplets):
    """
    Produce a flat list of face images in this order:
    [a1, p1, n1, a2, p2, n2, a3, p3, n3, ...]
    """
    images = []
    
    for triplet in triplets:
        (a, p, n) = triplet
        images.append(files[a])
        images.append(files[p])
        images.append(files[n])
        
    return images


def compute_triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper"""
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


# def get_train_op(total_loss, learning_rate, global_step, moving_average_decay):
#     loss_averages_op = add_loss_summaries(total_loss)
    
#     with tf.control_dependencies([loss_averages_op]):
#         opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
#         # opt = tf.compat.v1.train.AdagradOptimizer(learning_rate)
#         grads = opt.compute_gradients(total_loss, tf.compat.v1.global_variables())
        
#     apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

#     # Track the moving averages of all trainable variables.
#     variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
#     variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
#     with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
#         train_op = tf.no_op(name='train')
        
#     return train_op


def evaluate(model, sess, summary, pairs, lfw_paths, actual_issame, batch_size, n_folds):
    n_images = len(actual_issame) * 2
    assert len(lfw_paths) == n_images

    start_time = time.time()
    embs = predict_embeddings(model, sess, lfw_paths, batch_size)

    _, _, accuracy, val, val_std, far = lfw.evaluate(embs, actual_issame, n_folds=n_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time

    # Add validation loss and accuracy to summary
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    

def train(model, trainer, lr_placeholder):
    pairs = lfw.read_pairs(args.lfw_pairs)
    lfw_paths, actual_issame = lfw.get_paths(args.lfw_dir, pairs)

    summary_op = tf.summary.merge_all()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=30)
        summary_writer = tf.compat.v1.summary.FileWriter(args.log_dir, sess.graph)

        epoch = 0
        while epoch < args.epochs:
            step = sess.run(trainer.global_step, feed_dict=None)
            epoch = step // args.epoch_size

            print("------ Epoch %i ------" % (epoch+1))
            print("learning rate: %f" % sess.run(trainer.opt._lr, feed_dict={lr_placeholder: args.learning_rate}))

            people_epoch = utils.sample_people_for_epoch(args.data_dir, args.people_per_epoch, args.faces_per_person)
            people_batches = int(np.ceil(args.people_per_epoch / args.people_per_batch))
            
            #FIXME: I think people_per_batch is unnecessary. remove later
            images = []
            for batch in range(people_batches):
                people_batch = get_people_for_batch(people_epoch, batch, args.people_per_batch)
                files = sample_faces(args.data_dir, people_batch, args.faces_per_person)
                # images_batch = [read_image(file) for file in files]
                images.extend(files)

            embs = predict_embeddings(model, sess, images, args.batch_size)
            
            epoch_triplets = utils.select_triplets(embs, args.faces_per_person, args.alpha)
            
            triplets_images = get_triplets_images(images, epoch_triplets)

            batches = int(np.ceil(len(triplets_images) / args.batch_size))
            
            total_losses = []
            for batch in range(batches):
                start_time = time.time()
                start = int(batch * args.batch_size)
                end = start + args.batch_size
                triplets_batch = [read_image(file) for file in triplets_images[start:end]]

                feed_dict = {model.input_tensor: triplets_batch, lr_placeholder: args.learning_rate}
                total_loss_batch, _, step = sess.run([trainer.loss, trainer.train_op, trainer.global_step], feed_dict=feed_dict)
                
                duration = time.time() - start_time
                print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tStep %d' % 
                    (epoch+1, batch+1, batches, duration, total_loss_batch, step))
                
                total_losses.append(total_loss_batch)

            mean_loss = np.mean(total_losses)
            summary = tf.compat.v1.Summary()
            summary.value.add(tag="train/epoch_loss", simple_value=mean_loss)
            summary.value.add(tag="train/epoch_triplet_count", simple_value=len(epoch_triplets))

            if epoch % args.eval_every == 0 or epoch == args.epochs-1:
                evaluate(model, sess, summary, pairs, lfw_paths, actual_issame, args.batch_size, args.lfw_n_folds)

            summary_writer.add_summary(summary, step) # TODO: find meaning of add_summary
            utils.save_variables_and_metagraph(sess, saver, args.model_dir, args.model_name, step)
            summary_writer.flush()

def main():

    assert args.faces_per_person % 3 == 0, 'faces_per_person must be multiple of 3'
    assert args.batch_size % 3 == 0, 'batch_size must be multiple of 3'

    graph = tf.Graph()
    with graph.as_default():
        model = Model(args.model_url, graph, args.emb_size, preprocess_fn=preprocess)
        lr_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        triplets = tf.reshape(model.embeddings, [-1, 3, args.emb_size])
        anchor, positive, negative = tf.unstack(triplets, 3, 1)
        triplet_loss = compute_triplet_loss(anchor, positive, negative, args.alpha)

        # Calculate the total losses
        regularization_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        global_step = tf.Variable(0, trainable=False)

        decay_steps = args.lr_decay_epochs * args.epoch_size
        learning_rate = tf.train.exponential_decay(lr_placeholder, global_step, decay_steps, args.lr_decay_factor, staircase=True) 
        tf.summary.scalar('learning_rate', learning_rate)
        # train_op = get_train_op(total_loss, learning_rate, global_step, args.moving_average_decay)

        trainer = Trainer(total_loss, learning_rate, global_step, args.moving_average_decay)
        train(model, trainer, lr_placeholder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", help="Base model directory.", required=True)
    parser.add_argument("--model_name", help="Model name.", required=True)
    parser.add_argument('--gpu_memory_fraction', type=float, 
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    parser.add_argument("--people_per_epoch", help="Number of people per epoch.", default=500)
    parser.add_argument("--people_per_batch", help="Number of people per batch.", default=100)
    parser.add_argument("--faces_per_person", help="Faces per person.", default=6)
    parser.add_argument("--batch_size", help="Batch size.", default=90)
    parser.add_argument("--model_url", help="Tensor hub model url.", 
                        default="https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/3")
    parser.add_argument("--data_dir", help="Data directory.", 
                        default="/mnt/disks/data/datasets/vggface2/train_mtcnnpy_160/")
    parser.add_argument("--emb_size", help="Person embedding size.", default=512)
    parser.add_argument("--epochs", type=int, help="Number of epochs.", default=500)
    parser.add_argument("--epoch_size", type=int, help="Number of batches per epoch.", default=1000)
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.', default=0.01)
    parser.add_argument("--lr_decay_epochs", help="Number of epochs to decay the learning rate to another step.", default=5)
    parser.add_argument("--lr_decay_factor", help="Learning rate decay factor", default=0.96)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)

    parser.add_argument("--lfw_pairs", help="The file containing the pairs to use for validation.", 
                        default="/mnt/disks/data/datasets/lfw/raw_mtcnnpy_160/pairs.txt")
    parser.add_argument("--lfw_dir", help="Path to the data directory containing aligned face patches.", 
                        default="/mnt/disks/data/datasets/lfw/raw_mtcnnpy_160")
    parser.add_argument("--eval_every", help="Evaluate on LFW data every n epochs.", default=1)

    parser.add_argument("--log_dir", help="Log directory.", required=True)
    parser.add_argument("--alpha", help="Positive to negative triplet distance margin.", default=0.2)
    parser.add_argument("--lfw_n_folds", help="Number of folds to use for cross validation. Mainly used for testing.", 
                        default=10)

    args = parser.parse_args()
    main()    

    
    