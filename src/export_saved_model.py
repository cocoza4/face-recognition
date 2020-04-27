import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

import models

K.set_learning_phase(0)


def load_ckpt(ckpt_dir, model):
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
 
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return ckpt_manager

def main():
    
    model = models.create_model(args.backbone, args.embedding_size)
    load_ckpt(args.ckpt_dir, model)

    inp_spec = tf.TensorSpec(shape=[None, args.image_size, args.image_size, 3], dtype=tf.float32, name="inputs")
    
    @tf.function(input_signature=[inp_spec])
    def pred_fn(x):
        embeddings = tf.nn.l2_normalize(model(x, training=False), axis=-1)
        return {'embeddings': embeddings}

    tf.saved_model.save(model, args.output_dir, signatures=pred_fn.get_concrete_function(inp_spec))

    # add warmup data
    image_path = '../images/Adrien_Brody_0011.jpg'
    image = Image.open(image_path).resize((args.image_size, args.image_size))
    inputs = np.expand_dims(image, axis=0).astype(np.float32)

    assets_extra_path = os.path.join(args.output_dir, 'assets.extra')
    os.mkdir(assets_extra_path)
    
    warmup_reqs_path = os.path.join(assets_extra_path, 'tf_serving_warmup_requests')  
    with tf.io.TFRecordWriter(warmup_reqs_path) as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name='serving_default'),
            inputs={'inputs': tf.compat.v1.make_tensor_proto(inputs)}
        )
        log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())

    print('Model exported successfully to %s' % args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, help='Backbone name.', required=True)
    parser.add_argument('--ckpt_dir', type=str, help='Model checkpoints directory.', required=True)
    parser.add_argument('--output_dir', type=str, help='Model output directory.', required=True)
    parser.add_argument('--embedding_size', type=int, help='Embedding size.', required=True)
    parser.add_argument('--image_size', type=int, help='Image size.', default=112)

    args = parser.parse_args()
    main()
