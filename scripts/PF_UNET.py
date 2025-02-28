from tensorflow.keras.regularizers import l2 
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation, MaxPool2D, 
                                     Conv2DTranspose, Concatenate, Dropout, multiply, add)
from tensorflow.keras.models import Model

# Conv Block
def conv_block(inputs, num_filters, dropout_rate=0.1, l2_lambda=0.01):
    x = Conv2D(num_filters, 3, padding='same', kernel_regularizer=l2(l2_lambda))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(num_filters, 3, padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    return x

# Polarization Fusion Layer
def polarization_Fusion_block(inputs, num_combination_filters=3, l2_lambda=0.01):
    # Generate new feature maps from combinations of input channels
    polarization_Fusion = Conv2D(filters=num_combination_filters, kernel_size=(1, 1), padding='same', 
                                      kernel_regularizer=l2(l2_lambda))(inputs)
    polarization_Fusion = Activation('relu')(polarization_Fusion)
    
    # Concatenate original input channels with the combination feature maps
    output = Concatenate()([inputs, polarization_Fusion])
    
    return output

# Attention Block
def attention_block(skip_connection, gating_signal, num_filters, l2_lambda=0.01):
    # 1x1 convolution on skip connection
    theta_x = Conv2D(num_filters, (1, 1), padding='same', kernel_regularizer=l2(l2_lambda))(skip_connection)
    # 1x1 convolution on gating signal (up-sampled output)
    phi_g = Conv2D(num_filters, (1, 1), padding='same', kernel_regularizer=l2(l2_lambda))(gating_signal)
    
    # Add and apply ReLU activation
    add_xg = add([theta_x, phi_g])
    relu_xg = Activation('relu')(add_xg)
    
    # 1x1 convolution to compute attention coefficients
    psi = Conv2D(1, (1, 1), padding='same', kernel_regularizer=l2(l2_lambda))(relu_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    
    # Multiply attention coefficients with skip connection
    attn_coefficients = multiply([skip_connection, sigmoid_xg])
    return attn_coefficients

# Encoder Block
def encoder_block(inputs, num_filters, dropout_rate=0.1, l2_lambda=0.01):
    x = conv_block(inputs, num_filters, dropout_rate, l2_lambda)
    p = MaxPool2D((2, 2))(x)
    return x, p

# Decoder Block with Attention Gate
def decoder_block(inputs, skip, num_filters, dropout_rate=0.1, l2_lambda=0.01):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    # Apply attention to skip connection
    attn_skip = attention_block(skip, x, num_filters, l2_lambda)
    x = Concatenate()([x, attn_skip])
    x = conv_block(x, num_filters, dropout_rate, l2_lambda)
    return x

# U-Net Model with Attention
def unet_regression_model_with_attention(input_shape, dropout_rate=0.05, l2_lambda=0.01, num_combination_filters=3):
    inputs = Input(input_shape)
    
    # Apply the polarization Fusion layer
    x = polarization_Fusion_block(inputs, num_combination_filters=num_combination_filters, l2_lambda=l2_lambda)
    
    # Encoder path
    s1, p1 = encoder_block(x, 32, dropout_rate, l2_lambda) 
    s2, p2 = encoder_block(p1, 64, dropout_rate, l2_lambda) 
    s3, p3 = encoder_block(p2, 128, dropout_rate, l2_lambda)  
    # Bottleneck
    b1 = conv_block(p3, 256, dropout_rate, l2_lambda)
    
    # Decoder path with attention
    d1 = decoder_block(b1, s3, 128, dropout_rate, l2_lambda) 
    d2 = decoder_block(d1, s2, 64, dropout_rate, l2_lambda) 
    d3 = decoder_block(d2, s1, 32, dropout_rate, l2_lambda) 
    
    # Output layer for regression
    outputs = Conv2D(1, 1, padding='same', activation='relu', kernel_regularizer=l2(l2_lambda))(d3)
    
    model = Model(inputs, outputs, name='CHM_Regression_Unet_Attention')
    return model

# Create the U-Net model instance
input_shape = (32, 32, 3)  # Example input shape, adjust as needed
model = unet_regression_model_with_attention(input_shape)
