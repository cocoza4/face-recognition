import logging
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image

import utils
from evaluate import cfp


def preprocess(path):
    rgb = utils.read_rgb(path)
    img = (rgb - 127.5) * 0.0078125
    return img


def main():

    model = tf.saved_model.load(args.model_path)
    pred_fn = model.signatures[args.model_input_signature]

    def pred_batch_fn(inputs):
        preprocessed = tf.cast([preprocess(path) for path in inputs], tf.float32)
        return pred_fn(preprocessed)[args.model_ouput_signature]

    cfp_ff_dir = Path(args.cfp_ff_dir)
    cfp_fp_dir = Path(args.cfp_fp_dir)
    ff_fold_files = list(cfp_ff_dir.glob('ff*.csv'))
    fp_fold_files = list(cfp_fp_dir.glob('fp*.csv'))
    ff_mapping_file = cfp_ff_dir / 'pair_list_ff.csv'
    fp_mapping_file = cfp_fp_dir / 'pair_list_fp.csv'

    cfp_evaluator = cfp.CFPEvaluator(cfp_ff_dir, cfp_fp_dir, ff_fold_files, fp_fold_files, ff_mapping_file, 
                                    fp_mapping_file, args.embedding_size, args.batch_size, far_target=args.far)

    cfp_evaluator.evaluate_ff(pred_batch_fn)
    cfp_evaluator.evaluate_fp(pred_batch_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--model_path', type=str, help='TensorFlow model path.', required=True)
    parser.add_argument('--model_input_signature', type=str, help='Model input signature.', default='serving_default')
    parser.add_argument('--model_ouput_signature', type=str, help='Model output signature.', default='embeddings')
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=256)
    parser.add_argument('--image_size', type=int, help='Image size.', default=112)
    parser.add_argument('--embedding_size', type=int, help='Embedding size.', default=512)
    parser.add_argument('--far', type=float, help='Target FAR.', default=1e-3)
    parser.add_argument("--cfp_ff_dir", help="Path to the CFP Front Face data directory containing aligned face images.")
    parser.add_argument("--cfp_fp_dir", help="Path to the CFP Front Profile data directory containing aligned face images.")

    args = parser.parse_args()
    main()
