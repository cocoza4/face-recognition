import tensorflow as tf
# import tensorflow_hub as hub


class NormDense(tf.keras.layers.Layer):

    def __init__(self, classes=1000):
        super(NormDense, self).__init__()
        self.classes = classes

    def build(self, input_shape):
        self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
                                 initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        x = tf.matmul(inputs, norm_w)

        return x

class ArcFaceModel(tf.keras.Model):
    
    def __init__(self, backbone, embedding_size):
        super().__init__()
        self.backbone = backbone
        self.bn1 = tf.keras.layers.BatchNormalization()
        # self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense = tf.keras.layers.Dense(embedding_size, use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
        
    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.bn1(x, training=training)
        # x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.bn2(x, training=training)
        x = tf.nn.l2_normalize(x, axis=-1)
        return x


# class Model:

#     def __init__(self, url, graph, emb_size, preprocess_fn=None, height=160, width=160):
#         with graph.as_default():
#             module = hub.Module(url, trainable=True, tags={"train"})
#             self.emb_size = emb_size
#             self.input_tensor = tf.compat.v1.placeholder(tf.float32, [None, height, width, 3], name='input_images')
#             decoded_images = tf.map_fn(preprocess_fn, self.input_tensor, name='decoded_images')
#             features = module(decoded_images)
#             prelogits = tf.keras.layers.Dense(emb_size, activation=None)(features)
#             self.embeddings = tf.math.l2_normalize(prelogits, axis=1, name='embedding')

