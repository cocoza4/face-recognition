import os
import io
import itertools
import logging
import argparse
import contextlib2
from PIL import Image
from tqdm.auto import tqdm
from sklearn.utils import shuffle
import tensorflow as tf

import utils


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.

    Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

    Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards) for idx in range(num_shards)]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def create_tf_example(filename, label):

    with tf.io.gfile.GFile(filename, mode='rb') as f:
        encoded_jpg = f.read()
        image = Image.open(io.BytesIO(encoded_jpg))
        image_width, image_height = image.size

    feature_dict = {
        'image/width': int64_feature(image_width),
        'image/height': int64_feature(image_height),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/label': int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def create_tf_records(data_paths, output_path, n_shards):
    logging.info("creating tfrecords from %s of %d shards into %s", ', '.join(data_paths), n_shards, output_path)

    with contextlib2.ExitStack() as tf_record_close_stack:
        utils.create_dir(output_path)
        base_output_path = output_path + '/data.tfrecords'
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, base_output_path, n_shards)

        paths, labels = utils.combine_datasets(data_paths)
        paths, labels = shuffle(paths, labels)
        logging.info("%d images found" % len(paths))

        for i, (path, label) in tqdm(enumerate(zip(paths, labels))):
            tf_example = create_tf_example(path, label)
            shard_idx = i % n_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())               
            
def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", "--data_path", action="append", help="Source data path.", 
                        required=True, type=str)
    parser.add_argument("--output_path", help="Output data path.", required=True, type=str)
    parser.add_argument("--n_shards", help="Number of shards.", default=10, required=False, type=int)

    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    create_tf_records(args.data_path, args.output_path, args.n_shards)
