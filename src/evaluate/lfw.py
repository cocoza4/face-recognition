import os
from tqdm.auto import tqdm
import numpy as np
from scipy import interpolate
from sklearn.model_selection import KFold

from evaluate.eval_utils import calculate_accuracy, calculate_val_far_frr


class LFWEvaluator:

    def __init__(self, lfw_dir, lfw_pairs, batch_size, embedding_size, n_folds=10):
        pairs = read_pairs(lfw_pairs)
        lfw_paths, issame = get_paths(lfw_dir, pairs)

        self.image_paths = lfw_paths
        self.issame = issame
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.n_folds = n_folds

        self.n_images = len(self.issame) * 2
        assert len(self.image_paths) == self.n_images

    def evaluate(self, predict_fn):
        embs_array = np.zeros((self.n_images, self.embedding_size))
        it = tqdm(range(0, self.n_images, self.batch_size), 'Evaluate on LFW')
        for start in it:
            end = start + self.batch_size
            embs_array[start:end] = predict_fn(self.image_paths[start:end])
            
        _, _, accuracy, val, val_std, far, frr = evaluate(embs_array, self.issame, n_folds=self.n_folds)
        
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, FRR=%2.5f' % (val, val_std, far, frr))

        return np.mean(accuracy), val, far, frr

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list


def distance(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, n_folds=10, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=n_folds, shuffle=False)
    
    tprs = np.zeros((n_folds,nrof_thresholds))
    fprs = np.zeros((n_folds,nrof_thresholds))
    accuracy = np.zeros((n_folds))
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
            
        dist = distance(embeddings1-mean, embeddings2-mean)
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
        print('Best threshold for fold %d: %f' % (fold_idx, thresholds[best_threshold_index]))
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, n_folds=10, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=n_folds, shuffle=False)
    
    val = np.zeros(n_folds)
    far = np.zeros(n_folds)
    frr = np.zeros(n_folds)
    
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean)
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx], _ = calculate_val_far_frr(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        val[fold_idx], far[fold_idx], frr[fold_idx] = calculate_val_far_frr(threshold, dist[test_set], actual_issame[test_set])
  
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    frr_mean = np.mean(frr)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, frr_mean


def evaluate(embeddings, actual_issame, n_folds=10, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), n_folds=n_folds, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far, frr = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, n_folds=n_folds, subtract_mean=subtract_mean)

    return tpr, fpr, accuracy, val, val_std, far, frr