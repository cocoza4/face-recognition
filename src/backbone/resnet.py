import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, ZeroPadding2D, Dense, Reshape, PReLU, Flatten, GlobalAveragePooling2D


class AveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, x):
        return tf.reduce_mean(x, axis=[1, 2], keepdims=True)


class Bottleneck(tf.keras.Model):
    
    expansion = 4
    
    def __init__(self, filters, stage, block, strides=1, downsample=None):
        super().__init__()
        block_name = str(stage) + "_" + str(block)
        name = 'conv' + block_name
        
        self.conv1 = Conv2D(filters, kernel_size=1, strides=1, 
                            use_bias=False, name=name+'_conv1')
        self.bn1 = BatchNormalization(name=name+'_bn1')
        self.relu1 = PReLU(name=name+'_relu1')
        
        self.conv2 = Conv2D(filters, kernel_size=3, strides=strides, 
                            padding='same', use_bias=False, name=name+'_conv2')
        self.bn2 = BatchNormalization(name=name+'_bn2')
        self.relu2 = PReLU(name=name+'_relu2')
        
        self.conv3 = Conv2D(filters * self.expansion, kernel_size=1, strides=1, 
                            use_bias=False, name=name+'_conv3')
        self.bn3 = BatchNormalization(name=name+'_bn3')
        self.relu3 = PReLU(name=name+'_relu3')
        
        self.downsample = downsample
        
    def call(self, inputs, training=False):
        
        if self.downsample is not None:
            identity = self.downsample(inputs, training=training)
        else:
            identity = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.keras.layers.add([x, identity])
        x = self.relu3(x)
        
        return x
    
class BasicBlock(tf.keras.Model):
    
    expansion = 1
    
    def __init__(self, filters, stage, block, strides=1, downsample=None):
        super().__init__()
        block_name = str(stage) + "_" + str(block)
        name = 'conv' + block_name
        
        self.conv1 = Conv2D(filters, kernel_size=3, strides=strides, 
                            padding='same', use_bias=False, name=name+'_conv1')
        self.bn1 = BatchNormalization(name=name+'_bn1')
        self.relu1 = PReLU(name=name+'_relu1')
        
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, 
                            padding='same', use_bias=False, name=name+'_conv2')
        self.bn2 = BatchNormalization(name=name+'_bn2')
        self.relu2 = PReLU(name=name+'_relu2')
        
        self.downsample = downsample
        
    def call(self, inputs, training=False):
        
        if self.downsample is not None:
            identity = self.downsample(inputs, training=training)
        else:
            identity = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.keras.layers.add([x, identity])
        x = self.relu2(x)
        
        return x

class SEBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters, name='conv'):
        super().__init__()

        # Due to memory limitation, according to insightface, dense layers of squeeze and excite operations in SE are replaced by Conv layers. 
        # self.pool = GlobalAveragePooling2D(name=name+'_se_pool')
        # self.dense1 = Dense(filters // 16, activation='relu', name=name+'_se_fc1') # squeeze
        # self.dense2 = Dense(filters, activation='sigmoid', name=name+'_se_fc2') # excite
        self.pool = AveragePooling2D(name=name+'_se_pooling')
        
        self.conv1 = Conv2D(filters//16, kernel_size=(1, 1), strides=(1, 1), padding='valid',  # squeeze
                               use_bias=True, name=name + '_se_conv1')
        self.relu1 = PReLU(name=name + '_se_relu1')
        self.conv2 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='valid',  # excite
                               use_bias=True, name=name + '_se_conv2')
        self.sigmoid = Activation('sigmoid', name=name + '_se_sigmoid')
        # self.reshape = Reshape([1, 1, filters])
        
    def call(self, inputs):
        x = self.pool(inputs)
        # x = self.dense1(x)
        # x = self.dense2(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        # x = self.reshape(x)
        x = tf.keras.layers.multiply([x, inputs])
        return x

class SEBottleneck(Bottleneck):

    def __init__(self, filters, stage, block, strides=1, downsample=None):
        super().__init__(filters, stage, block, strides=strides, downsample=downsample)
        block_name = str(stage) + "_" + str(block)
        name = 'seconv' + block_name
        self.se_block = SEBlock(filters * self.expansion, name=name)

    def call(self, inputs, training=False):
        
        if self.downsample is not None:
            identity = self.downsample(inputs, training=training)
        else:
            identity = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.se_block(x, training=training)
        x = tf.keras.layers.add([x, identity])
        x = self.relu3(x)
        
        return x


class SEBasicBlock(BasicBlock):

    def __init__(self, filters, stage, block, strides=1, downsample=None):
        super().__init__(filters, stage, block, strides=strides, downsample=downsample)
        block_name = str(stage) + "_" + str(block)
        name = 'seconv' + block_name
        self.se_block = SEBlock(filters, name=name)

    def call(self, inputs, training=False):
        if self.downsample is not None:
            identity = self.downsample(inputs, training=training)
        else:
            identity = inputs
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.se_block(x, training=training)
        x = tf.keras.layers.add([x, identity])
        x = self.relu2(x)
        
        return x
        

class ResNet(tf.keras.Model):
    
    def __init__(self, blocks, name='resnet', block_type=BasicBlock):
        super().__init__()
        
        self.conv1 = Conv2D(64, kernel_size=3, strides=1, 
                            padding='valid', use_bias=False, name='conv1')
        self.bn1 = BatchNormalization(name='conv1_bn')
        self.relu1 = PReLU(name='conv1_relu')
        
        self.stage2 = self.build_stage(64, block_type, blocks[0], strides=1, stage=2)
        self.stage3 = self.build_stage(128, block_type, blocks[1], strides=2, stage=3)
        self.stage4 = self.build_stage(256, block_type, blocks[2], strides=2, stage=4)
        self.stage5 = self.build_stage(512, block_type, blocks[3], strides=2, stage=5)

    def build_stage(self, filters, block_type, n_blocks, strides, stage):
        
        downsample = tf.keras.Sequential([
            Conv2D(filters * block_type.expansion, kernel_size=1, strides=strides, name=f'conv{stage}_0_conv0'),
            BatchNormalization(name=f'conv{stage}_0_bn0')
        ])
        
        blocks = [block_type(filters, stage, block=1, strides=strides, downsample=downsample)]
        for i in range(1, n_blocks):
            blocks.append(block_type(filters, stage, block=i+1, strides=1))
            
        return tf.keras.Sequential(blocks)

    def call(self, inputs, training=False):
        x = ZeroPadding2D()(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)
        
        return x
    

class SEResNet(ResNet):
    
    def __init__(self, blocks, name='seresnet', block_type=SEBasicBlock):
        super().__init__(blocks, name, block_type)
        
