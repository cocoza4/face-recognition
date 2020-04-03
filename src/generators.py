import os
import tensorflow as tf

import utils

class DataGenerator:

    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        self.paths, self.labels = utils.get_data(data_dir)
        self.n_classes = len(self.paths)
        assert len(self.paths) == len(self.labels)

    def steps_per_epoch(self):
        ds = (tf.data.Dataset.from_tensor_slices(self.paths)
                .batch(self.batch_size))
        steps = 0
        for _ in ds:
            steps += 1
        return steps

    def generate(self, preprocess_fn=None, shuffle_buffer=200000):
        ds = (tf.data.Dataset.from_tensor_slices((self.paths, self.labels))
            .cache()
            .shuffle(shuffle_buffer)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

        if preprocess_fn:
            def p(inputs, labels): return preprocess_fn(inputs), labels
            ds = ds.map(p, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(self.batch_size)
        return ds


class TFRecordDataGenerator:
    
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def _tfrecord_files(self):
        return [f'{self.data_dir}/{f}' for f in tf.io.gfile.listdir(self.data_dir)]

    def steps_per_epoch(self):
        tfrecords = self._tfrecord_files()
        ds = (tf.data.TFRecordDataset(filenames=tfrecords)
                .batch(self.batch_size))
        steps = 0
        for _ in ds:
            steps += 1
        return steps
    
    def generate(self, example_parser, preprocess_fn=None, shuffle_buffer=100000):
        tfrecords = self._tfrecord_files()

        ds = (tf.data.TFRecordDataset(filenames=tfrecords)
            .shuffle(shuffle_buffer)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            .map(example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE))

        if preprocess_fn:
            ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(self.batch_size).repeat()
        return ds
