from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *
from keras import backend as K
import numpy as np
import keras, math
from .keras_layer_L2Normalization import L2Normalization

def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), trainable=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1',
               trainable=trainable)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1,dila=(1,1), trainable=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        dilation_rate=dila,
                        name='conv_dw_%d' % block_id,
                        trainable=trainable)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id,
               trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

# Original MobileNet from paper.
def nn_p2p3(img_input=None, alpha=1.0, depth_multiplier=1,  trainable=True):
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=False)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, trainable=False)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, trainable=trainable)
    stage2 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,trainable=trainable)

    x = _depthwise_conv_block(stage2, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, trainable=trainable)
    stage3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,trainable=trainable)
    # print('stage3:', stage3._keras_shape[1:])

    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])

    P2_up = L2Normalization(gamma_init=10, name='P2norm')(stage2)
    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    conc = Concatenate(axis=-1)([P2_up, P3_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=-1, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    return [x_class, x_regr]

def nn_p3p4(img_input=None, alpha=1.0, depth_multiplier=1,  trainable=True):
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=False)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, trainable=False)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, trainable=trainable)
    stage2 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,trainable=trainable)

    x = _depthwise_conv_block(stage2, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, trainable=trainable)
    stage3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,trainable=trainable)
    # print('stage3:', stage3._keras_shape[1:])

    x = _depthwise_conv_block(stage3, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, trainable=trainable)
    stage4 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, trainable=trainable)
    # print('stage4:', stage4._keras_shape[1:])

    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])

    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=-1, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    return [x_class, x_regr]

def nn_p4p5(img_input=None, alpha=1.0, depth_multiplier=1,  trainable=True):
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=False)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, trainable=False)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, trainable=trainable)
    stage2 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,trainable=trainable)

    x = _depthwise_conv_block(stage2, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, trainable=trainable)
    stage3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,trainable=trainable)
    # print('stage3:', stage3._keras_shape[1:])

    x = _depthwise_conv_block(stage3, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, trainable=trainable)
    stage4 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, trainable=trainable)
    # print('stage4:', stage4._keras_shape[1:])

    x = _depthwise_conv_block(stage4, 1024, alpha, depth_multiplier,
                              strides=(1, 1), block_id=12, trainable=trainable)
    stage5 = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, trainable=trainable)
    # print('stage5:', stage5._keras_shape[1:])

    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])
    P5_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(stage5)
    # print('P5_up: ', P5_up._keras_shape[1:])

    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=-1, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    return [x_class, x_regr]

def nn_p2p3p4(img_input=None, alpha=1.0, depth_multiplier=1,  trainable=True):
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=False)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, trainable=False)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, trainable=trainable)
    stage2 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,trainable=trainable)

    x = _depthwise_conv_block(stage2, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, trainable=trainable)
    stage3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,trainable=trainable)
    # print('stage3:', stage3._keras_shape[1:])

    x = _depthwise_conv_block(stage3, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, trainable=trainable)
    stage4 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, trainable=trainable)
    # print('stage4:', stage4._keras_shape[1:])

    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])

    P2_up = L2Normalization(gamma_init=10, name='P2norm')(stage2)
    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    conc = Concatenate(axis=-1)([P2_up, P3_up, P4_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=-1, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    return [x_class, x_regr]

def nn_p3p4p5(img_input=None, alpha=1.0, depth_multiplier=1,  trainable=True):
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=False)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, trainable=False)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, trainable=trainable)
    stage2 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,trainable=trainable)

    x = _depthwise_conv_block(stage2, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, trainable=trainable)
    stage3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,trainable=trainable)
    # print('stage3:', stage3._keras_shape[1:])

    x = _depthwise_conv_block(stage3, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, trainable=trainable)
    stage4 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, trainable=trainable)
    # print('stage4:', stage4._keras_shape[1:])

    x = _depthwise_conv_block(stage4, 1024, alpha, depth_multiplier,
                              strides=(1, 1), block_id=12, trainable=trainable)
    stage5 = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, trainable=trainable)
    # print('stage5:', stage5._keras_shape[1:])

    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])
    P5_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(stage5)
    # print('P5_up: ', P5_up._keras_shape[1:])

    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=-1, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    return [x_class, x_regr]

def nn_p2p3p4p5(img_input=None, alpha=1.0, depth_multiplier=1,  trainable=True):
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), trainable=False)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, trainable=False)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, trainable=trainable)
    stage2 = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,trainable=trainable)

    x = _depthwise_conv_block(stage2, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, trainable=trainable)
    stage3 = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,trainable=trainable)
    # print('stage3:', stage3._keras_shape[1:])

    x = _depthwise_conv_block(stage3, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, trainable=trainable)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, trainable=trainable)
    stage4 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, trainable=trainable)
    # print('stage4:', stage4._keras_shape[1:])

    x = _depthwise_conv_block(stage4, 1024, alpha, depth_multiplier,
                              strides=(1, 1), block_id=12, trainable=trainable)
    stage5 = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, trainable=trainable)
    # print('stage5:', stage5._keras_shape[1:])

    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(stage3)
    # print('P3_up: ', P3_up._keras_shape[1:])
    P4_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P4up', trainable=trainable)(stage4)
    # print('P4_up: ', P4_up._keras_shape[1:])
    P5_up = Deconvolution2D(256, kernel_size=4, strides=4, padding='same',
                            kernel_initializer='glorot_normal', name='P5up', trainable=trainable)(stage5)
    # print('P5_up: ', P5_up._keras_shape[1:])

    P2_up = L2Normalization(gamma_init=10, name='P2norm')(stage2)
    P3_up = L2Normalization(gamma_init=10, name='P3norm')(P3_up)
    P4_up = L2Normalization(gamma_init=10, name='P4norm')(P4_up)
    P5_up = L2Normalization(gamma_init=10, name='P5norm')(P5_up)
    conc = Concatenate(axis=-1)([P2_up, P3_up, P4_up, P5_up])

    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(conc)
    feat = BatchNormalization(axis=-1, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(1, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    return [x_class, x_regr]

# focal loss like
def prior_probability_onecls(num_class=1, probability=0.01):
	def f(shape, dtype=keras.backend.floatx()):
		assert(shape[0] % num_class == 0)
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		# set bias to -log(p/(1 - p)) for background
		return result
	return f
