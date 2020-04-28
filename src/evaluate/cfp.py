import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from scipy import interpolate

from evaluate.eval_utils import calculate_accuracy, calculate_val_far_frr


class CFPEvaluator:

    def __init__(self, front_dir, profile_dir, ff_fold_files, fp_fold_files, ff_mapping_file, 
                fp_mapping_file, embedding_size, batch_size, far_target=1e-3):
        self.front_dir = Path(front_dir)
        self.profile_dir = Path(profile_dir)
        self.ff_fold_files = ff_fold_files
        self.fp_fold_files = fp_fold_files
        self.ff_mapping = get_mapping(ff_mapping_file)
        self.fp_mapping = get_mapping(fp_mapping_file)
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.far_target = far_target

    def evaluate_ff(self, predict_fn):
        folds, issame_folds = get_folds(self.ff_fold_files, self.front_dir, self.front_dir, self.ff_mapping, 
                                        self.ff_mapping, predict_fn, self.embedding_size, self.batch_size)
        
        thresholds = np.arange(0, 4, 0.01)
        tpr, fpr, accuracy = calculate_roc(thresholds, folds, issame_folds)
        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far, frr = calculate_val(thresholds, folds, issame_folds, far_target=self.far_target)

        print('[cfp_ff]Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('[cfp_ff]Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, FRR=%2.5f' % (val, val_std, far, frr))
        
        return np.mean(accuracy), val, far, frr

    def evaluate_fp(self, predict_fn):
        folds, issame_folds = get_folds(self.fp_fold_files, self.front_dir, self.profile_dir, self.ff_mapping, 
                                        self.fp_mapping, predict_fn, self.embedding_size, self.batch_size)
            
        thresholds = np.arange(0, 4, 0.01)
        tpr, fpr, accuracy = calculate_roc(thresholds, folds, issame_folds)
        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far, frr = calculate_val(thresholds, folds, issame_folds, far_target=self.far_target)

        print('[cfp_fp]Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        print('[cfp_fp]Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, FRR=%2.5f' % (val, val_std, far, frr))
        
        return np.mean(accuracy), val, far, frr

def get_folds(fold_files, id1_dir, id2_dir, id1_mapping, id2_mapping, predict_fn, embedding_size, batch_size):
    folds = []
    issame_folds = []
    
    for i, f in enumerate(fold_files):
        df_fold = pd.read_csv(f)
        df_fold['id1'] = df_fold['id1'].map(lambda x: str(id1_dir/id1_mapping[x]))
        df_fold['id2'] = df_fold['id2'].map(lambda x: str(id2_dir/id2_mapping[x]))
        paths, same_list = get_XY(df_fold)
        embs = predict_embeddings(predict_fn, paths, embedding_size, batch_size, i+1)
        folds.append(embs)
        issame_folds.append(same_list)

    return folds, issame_folds


def predict_embeddings(predict_fn, paths, embedding_size, batch_size, fold):
    n_images = len(paths)
    embs_array = np.zeros((n_images, embedding_size))
    it = tqdm(range(0, n_images, batch_size), 'Evaluate on CFP, fold {}'.format(fold))
    for start in it:
        end = start + batch_size
        embs_array[start:end] = predict_fn(paths[start:end])
        
    return embs_array

def distance(embs):
    embs1 = embs[0::2]
    embs2 = embs[1::2]
    diff = np.subtract(embs1, embs2)
    dist = np.sum(np.square(diff), 1)
    return dist

def calculate_roc(thresholds, folds, issame_folds):
    n_folds = len(folds)
    n_thresholds = len(thresholds)
    acc_train = np.zeros((n_thresholds))

    tprs = np.zeros((n_folds, n_thresholds))
    fprs = np.zeros((n_folds, n_thresholds))
    accuracy = np.zeros((n_folds))
    
    for fold_idx in range(len(folds)):
        embs_train = np.concatenate([folds[j] for j in range(n_folds) if fold_idx != j], axis=0)
        same_train = np.concatenate([issame_folds[j] for j in range(n_folds) if fold_idx != j], axis=0)
        dist_train = distance(embs_train)

        embs_eval = folds[fold_idx]
        same_eval = issame_folds[fold_idx]
        dist_eval = distance(embs_eval)

        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist_train, same_train)

        best_threshold_idx = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                            calculate_accuracy(threshold, dist_eval, same_eval)
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_idx], dist_eval, same_eval)

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        
    return tpr, fpr, accuracy

def calculate_val(thresholds, folds, issame_folds, far_target):
    n_folds = len(folds)
    val = np.zeros(n_folds)
    far = np.zeros(n_folds)
    frr = np.zeros(n_folds)
    n_thresholds = len(thresholds)
    for fold_idx in range(len(folds)):
        embs_train = np.concatenate([folds[j] for j in range(n_folds) if fold_idx != j], axis=0)
        same_train = np.concatenate([issame_folds[j] for j in range(n_folds) if fold_idx != j], axis=0)
        dist_train = distance(embs_train)

        embs_eval = folds[fold_idx]
        same_eval = issame_folds[fold_idx]
        dist_eval = distance(embs_eval)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(n_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx], _ = calculate_val_far_frr(threshold, dist_train, same_train)
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0

            val[fold_idx], far[fold_idx], frr[fold_idx] = \
                    calculate_val_far_frr(threshold, dist_eval, same_eval)

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    frr_mean = np.mean(frr)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, frr_mean

def get_mapping(f, header=None):
    df = pd.read_csv(f, header=header)
    mapping = {row[0]: row[1] for _, row in df.iterrows()}
    return mapping

def get_XY(df):
    image_paths = []
    same_list = []
    for _, row in df.iterrows():
        id1 = row['id1']
        id2 = row['id2']
        same = bool(row['same'])
        
        if not os.path.exists(id1) or not os.path.exists(id2):
            pass
            # print('Skipping {} and {} pair, either one of them or both are missing '
            #       'due to unsuccessful face detection'.format(str(id1), str(id2)))
        else:
            image_paths += [id1, id2]
            same_list.append(same)
        
    return image_paths, same_list
    