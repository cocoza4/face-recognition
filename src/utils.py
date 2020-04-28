import os
import yaml
import time
import random
import itertools
import shutil
import numpy as np
from PIL import Image


class ImageClass:
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def read_rgb(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    return image
    
def get_image_classes(path):
    dataset = []
    classes = [o for o in os.listdir(path) \
                    if os.path.isdir(os.path.join(path, o))]
    classes.sort()
    nrof_classes = len(classes)
    for class_name in classes:
        facedir = os.path.join(path, class_name)
        image_paths = get_files_under_directory(facedir, include_parent=True)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def flatten(xs):
    return list(itertools.chain(*xs))

def create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_configs(path, config):
    with open(path, 'w') as f:
        f.write('Parameters\n')
        for arg, value in config.items():
            f.write('{}: {}\n'.format(arg, value))

def load_config_file(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        return config

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

    return flatten(paths), flatten(labels)

def combine_datasets(data_paths):
    all_paths, all_labels = [], []
    id_start = 0
    for path in data_paths:
        paths, labels = get_data(path)
        labels = np.array(labels) + id_start
        all_paths.extend(paths)
        all_labels.extend(labels)

        id_start += np.max(labels) + 1
    return all_paths, all_labels

def get_files_under_directory(path, include_parent=False):
    files = [os.path.join(path, name) if include_parent else name \
             for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    return files

def to_rgb(img):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
        
    w, h, c = img.shape
    if c == 1:
        rgb = np.empty((w, h, 3), dtype=np.uint8)
        rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = img.squeeze()
    elif c > 3:
        rgb = img[:, :, 0:3]
    else:
        rgb = img
        
    return rgb

def sample_people_for_epoch(dataset_root, people_per_epoch, faces_per_person):
    """
    Return a sampled list of people_per_epoch people who has more than 
    faces_per_person images in the dataset_root. Note that the list contains the
    names (or folders' names), not full path to them.
    """
    # sample some candidates with a margin first, so we can avoid scanning the
    # entire dataset
    margin = 1.2
    candidates = random.sample(os.listdir(dataset_root), int(margin * people_per_epoch))
    qualified_people = []
    
    # ensure each candidate has at least faces_per_person images
    for name in candidates:
        name_path = os.path.join(dataset_root, name)
        
        if not os.path.isfile(name_path):
            # this is a folder containing images of a person
            files = get_files_under_directory(name_path)
            
            if len(files) >= faces_per_person:
                qualified_people.append(name)
            
        if len(qualified_people) >= people_per_epoch:
            break
            
    return qualified_people


def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    """
    Save the model checkpoint, and meta graph (if none exists).
    """
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    
    # only save meta graph the first time
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
        
    # summary = tf.compat.v1.Summary()
    # #pylint: disable=maybe-no-member
    # summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    # summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    # summary_writer.add_summary(summary, step)


def select_triplets(embeddings, tot_images_per_person, alpha):
    """
    Choosing good triplets is crucial and should strike a balance between
    selecting informative (i.e. challenging) examples and swamping training with examples that
    are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    the image n at random, but only between the ones that violate the triplet loss margin. The
    latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    choosing the maximally violating example, as often done in structured output learning.
    
    Params:
        embeddings  A np array of shape (batch_size, emb_size) where embeddings at index 
                    [0:tot_images_per_person, :]                              belong to person 1
                    [tot_images_per_person:2 * tot_images_per_person, :]      belong to person 2
                    [2 * tot_images_per_person:3 * tot_images_per_person, :]  belong to person 3
                    ...
        tot_images_per_person   Total number of embeddings per person
        alpha
    Returns
        triplets    A list of [(a, p, n), (a, p, n), ...] where a, p, n are indices of 
                    embeddings array.
    """
    tot_embeddings, emb_size = np.shape(embeddings)
    tot_people = int(np.ceil(tot_embeddings / tot_images_per_person))
    triplets = []
    # distances = []

    # compute triplets for each person
    for person in range(tot_people):
        idx_person_start = person * tot_images_per_person
        idx_person_end = min(idx_person_start + tot_images_per_person, tot_embeddings)
        #print("====== %i - %i ======" % (idx_person_start, idx_person_end))

        # set each image of a person to be an anchor
        for idx_person_a in range(idx_person_start, idx_person_end):
            dist_sqr_all = np.sum(np.square(embeddings[idx_person_a] - embeddings), 1)
            #print(dist_sqr_all)

            # for every posible positive pair, randomly get a hard negative pair
            for idx_person_p in range(idx_person_a + 1, idx_person_end):
                dist_sqr_p = dist_sqr_all[idx_person_p]
                dist_sqr_n = np.copy(dist_sqr_all)
                dist_sqr_n[idx_person_start: idx_person_end] = 1000000      # something big to exlude p pairs
                all_hard_n = np.where(dist_sqr_n - dist_sqr_p < alpha)      # these are the hard n pair
                n_count = np.shape(all_hard_n)[1]
                #print("idx_person_p %i, dist_sqr_p %.2f" % (idx_person_p, dist_sqr_p))
                #print(dist_sqr_all)
                #print(dist_sqr_n)
                #print(all_hard_n)

                if n_count > 0:
                    # if there is at least 1 hard n pair
                    idx_all_hard_n = np.random.randint(0, n_count)
                    idx_person_n = all_hard_n[0][idx_all_hard_n]
                    triplets.append((idx_person_a, idx_person_p, idx_person_n))
                    # distances.append((dist_sqr_all[idx_person_p], dist_sqr_all[idx_person_n]))
                    #print("diff %.2f, alpha %f" % (dist_sqr_all[idx_person_n] - dist_sqr_all[idx_person_p], alpha))
                else:
                    # no hard n pair. Skip
                    pass
                

    np.random.shuffle(triplets)
    return triplets
