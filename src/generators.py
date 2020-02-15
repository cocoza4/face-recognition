import os
import tensorflow as tf

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


class DataGenerator:

    def __init__(self, data_dir, batch_size):
        self.batch_size = batch_size
        paths, labels = get_data(data_dir)
        self.n_classes = len(paths)
        self.paths = [path for cls in paths for path in cls]
        self.labels = [label for cls in labels for label in cls]
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