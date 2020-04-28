import logging
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

import utils
from evaluate import lfw


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

    lfw_evaluator = lfw.LFWEvaluator(args.lfw_dir, args.lfw_pairs, 
                                    batch_size=args.batch_size, embedding_size=args.embedding_size, 
                                    far_target=args.far, n_folds=args.lfw_n_folds)

    lfw_evaluator.evaluate(pred_batch_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--model_path', type=str, help='TensorFlow model path.', required=True)
    parser.add_argument('--model_input_signature', type=str, help='Model input signature.', default='serving_default')
    parser.add_argument('--model_ouput_signature', type=str, help='Model output signature.', default='embeddings')
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=256)
    parser.add_argument('--image_size', type=int, help='Image size.', default=112)
    parser.add_argument('--embedding_size', type=int, help='Embedding size.', default=512)
    parser.add_argument('--far', type=float, help='Target FAR.', default=1e-3)
    parser.add_argument("--lfw_pairs", help="The file containing the pairs to use for validation.")
    parser.add_argument("--lfw_dir", help="Path to the data directory containing aligned face images.")
    parser.add_argument("--lfw_n_folds", help="Number of folds to use for cross validation. Mainly used for testing.", default=10)

    args = parser.parse_args()
    main()
