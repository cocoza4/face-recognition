import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten


class ArcFaceModel(tf.keras.Model):
    
    def __init__(self, backbone, embedding_size):
        super().__init__()
        self.backbone = backbone
        self.bn1 = BatchNormalization()
        # self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense = Dense(embedding_size, use_bias=True)
        self.bn2 = BatchNormalization(scale=False)
        
    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)
        x = self.bn1(x, training=training)
        x = Flatten()(x)
        x = self.dense(x)
        x = self.bn2(x, training=training)
        return x


def create_model(backbone_name, emb_size):
    if backbone_name == 'densenet121':
        backbone = tf.keras.applications.DenseNet121(weights=None, include_top=False, pooling='avg')
    elif backbone_name == 'se-resnet50':
        from backbone.resnet import SEResNet
        backbone = SEResNet(blocks=[3, 4, 6, 3])

    model = ArcFaceModel(backbone, emb_size)
    return model

