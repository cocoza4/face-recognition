import tensorflow as tf


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
        x = self.dense(x)
        x = self.bn2(x, training=training)
        return x


def create_model(backbone_name, emb_size):
    if backbone_name == 'densenet121':
        backbone = tf.keras.applications.DenseNet121(weights=None, include_top=False, pooling='avg')
    elif backbone_name == 'resnet100':
        from ..backbone.resnet import ResNet100
        # backbond = ResNet100
        # TODO: implement me

    model = ArcFaceModel(backbone, emb_size)
    return model

