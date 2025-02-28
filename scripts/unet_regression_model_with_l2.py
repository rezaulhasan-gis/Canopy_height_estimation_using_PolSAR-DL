from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Dropout
from tensorflow.keras.models import Model

# Conv Block
def conv_block(inputs, num_filters, dropout_rate=0.2, l2_lambda=0.01):
    x = Conv2D(num_filters, 3, padding='same', kernel_regularizer=l2(l2_lambda))(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(num_filters, 3, padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(dropout_rate)(x)
    return x

# Encoder Block
def encoder_block(inputs, num_filters, dropout_rate=0.2, l2_lambda=0.01):
    x = conv_block(inputs, num_filters, dropout_rate, l2_lambda)
    p = MaxPool2D((2, 2))(x)
    return x, p

# Decoder Block
def decoder_block(inputs, skip, num_filters, dropout_rate=0.2, l2_lambda=0.01):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters, dropout_rate, l2_lambda)
    return x

# U-Net with Dropout and L2 regularization for CHM Regression
def unet_regression_model_with_l2(input_shape, dropout_rate=0.2, l2_lambda=0.01):
    inputs = Input(input_shape)
    """ Encoder """
    s1, p1 = encoder_block(inputs, 64, dropout_rate, l2_lambda)
    s2, p2 = encoder_block(p1, 128, dropout_rate, l2_lambda)
    s3, p3 = encoder_block(p2, 256, dropout_rate, l2_lambda) 
   
    """ Bridge """
    b1 = conv_block(p3, 256, dropout_rate, l2_lambda)
    
    """ Decoder """
    d1 = decoder_block(b1, s3, 256, dropout_rate, l2_lambda)
    d2 = decoder_block(d1, s2, 128, dropout_rate, l2_lambda)
    d3 = decoder_block(d2, s1, 64, dropout_rate, l2_lambda)
    
    """ Output """
    outputs = Conv2D(1, 1, padding='same', activation='relu', kernel_regularizer=l2(l2_lambda))(d3)
    
    model = Model(inputs, outputs, name='CHM_Regression_Unet_L2')
    return model
