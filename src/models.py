import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten

from backbone import resnet


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

    # ResNet
    if backbone_name == 'resnet18':
        backbone = resnet.ResNet(blocks=[2, 2, 2, 2], name=backbone_name)
    elif backbone_name == 'resnet34':
        backbone = resnet.ResNet(blocks=[3, 4, 6, 3], name=backbone_name)
    elif backbone_name == 'resnet50':
        backbone = resnet.ResNet(blocks=[3, 4, 6, 3], name=backbone_name, block_type=resnet.Bottleneck)
    elif backbone_name == 'resnet100':
        backbone = resnet.ResNet(blocks=[3, 4, 23, 3], name=backbone_name, block_type=resnet.Bottleneck)
    elif backbone_name == 'resnet152':
        backbone = resnet.ResNet(blocks=[3, 8, 36, 3], name=backbone_name, block_type=resnet.Bottleneck)

    # SE-ResNet
    elif backbone_name == 'seresnet18':
        backbone = resnet.SEResNet(blocks=[2, 2, 2, 2], name=backbone_name)
    elif backbone_name == 'seresnet34':
        backbone = resnet.SEResNet(blocks=[3, 4, 6, 3], name=backbone_name)
    elif backbone_name == 'seresnet50':
        backbone = resnet.SEResNet(blocks=[3, 4, 6, 3], name=backbone_name, block_type=resnet.SEBottleneck)
    elif backbone_name == 'seresnet100':
        backbone = resnet.SEResNet(blocks=[3, 4, 23, 3], name=backbone_name, block_type=resnet.SEBottleneck)
    elif backbone_name == 'seresnet152':
        backbone = resnet.SEResNet(blocks=[3, 8, 36, 3], name=backbone_name, block_type=resnet.SEBottleneck)

    # TODO: DenseNet

    model = ArcFaceModel(backbone, emb_size)
    return model

