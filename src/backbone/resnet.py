import tensorflow as tf
from tensorflow.keras import layers

class AveragePooling2D(layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, x):
        return tf.reduce_mean(x, axis=[1, 2], keepdims=True)


class BaseResidualBlock(layers.Layer):

    def __init__(self, filters, strides, name):
        super().__init__()
        self.bn1 = layers.BatchNormalization(name=name + '_bn1')
        self.conv1 = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', 
                                    use_bias=False, name=name + '_conv1')
        self.bn2 = layers.BatchNormalization(name=name + '_bn2')
        self.prelu1 = layers.PReLU(name=name + 'act1')
        self.conv2 = layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='valid',
                                    use_bias=False, name=name + '_conv2')
        self.bn3 = layers.BatchNormalization(name=name + '_bn3')

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = layers.ZeroPadding2D((1, 1))(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = self.prelu1(x)
        x = layers.ZeroPadding2D((1, 1))(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        return x


class ResidualBlock(BaseResidualBlock):
    
    def __init__(self, filters, strides, name, shortcut=True):
        super().__init__(filters, strides, name)

        self.shortcut = shortcut
        if shortcut:
            self.conv1_sc = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='valid',
                                            use_bias=False, name=name + '_conv1sc')
            self.bn_sc = layers.BatchNormalization(name=name + '_sc')
        
        
    def call(self, inputs, training=False):
        # Residual Block
        x = super().call(inputs, training=training)

        if self.shortcut:
            shortcut = self.bn_sc(self.conv1_sc(inputs))
        else:
            shortcut = inputs
        
        return x + shortcut


class SEResidualBlock(BaseResidualBlock):

    def __init__(self, filters, strides, name, shortcut=True):
        super().__init__(filters, strides, name)
        self.se_pool = AveragePooling2D(name=name + '_se_pooling')
        
        self.se_conv1 = layers.Conv2D(filters//16, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                        use_bias=True, name=name + '_se_conv1')
        self.se_prelu = layers.PReLU(name=name + '_se_act1')
        self.se_conv2 = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                    use_bias=True, name=name + '_se_conv2')
        self.se_sigmoid = layers.Activation('sigmoid', name=name + '_se_sigmoid')
        self.shortcut = shortcut
        if shortcut:
            self.conv1_sc = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='valid',
                                            use_bias=False, name=name + '_conv1sc')
            self.bn_sc = layers.BatchNormalization(name=name + '_sc')
    
    def call(self, inputs, training=False):
        bn3 = super().call(inputs, training=training)

        # SE
        x = self.se_pool(bn3)
        x = self.se_conv1(x)
        x = self.se_prelu(x)
        x = self.se_conv2(x)
        x = self.se_sigmoid(x)
        bn3 = tf.keras.layers.multiply([bn3, x])
        
        if self.shortcut:
            shortcut = self.bn_sc(self.conv1_sc(inputs))
        else:
            shortcut = inputs
        
        return bn3 + shortcut


class StackedBlock():
    
    def __init__(self, filters, blocks, strides, name):
        self.blocks = []
        self.block = SEResidualBlock(filters, strides=strides, name=name, shortcut=True)
        for i in range(blocks):
            block = SEResidualBlock(filters, strides=(1, 1), name=name+f'_block{i+1}', shortcut=False)
            self.blocks.append(block)
    
    def __call__(self, x, training=False):
        x = self.block(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x
    
        
class ResNet100(tf.keras.Model):
    
    def __init__(self, filters, blocks, embedding_size=512, name='resnet100'):
        super().__init__()
        assert len(filters) == len(blocks)
        
        self.pad = layers.ZeroPadding2D((1, 1), name='pad')
        self.conv1 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), 
                                    padding='valid', use_bias=False, name='conv1')
        self.bn1 = layers.BatchNormalization(name='conv1_bn')
        self.prelu1 = layers.PReLU(name='conv1_relu')
        
        self.stacks = []
        for i, f in enumerate(filters):
            name = f'conv{i+1}'
            stack = StackedBlock(f, blocks=blocks[i], strides=(2, 2), name=name)
            self.stacks.append(stack)
            
        self.bnx = layers.BatchNormalization()
        # self.dropout = tf.keras.layers.Dropout(0.4)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(embedding_size, use_bias=False)
        
        self.bny = layers.BatchNormalization(scale=False)
        
    def call(self, inputs, training=False):
        x = self.pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        
        for stack in self.stacks:
            x = stack(x)
        
        # TODO: later move this else where
        x = self.bnx(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.bny(x, training=training)
        
        return x
                               