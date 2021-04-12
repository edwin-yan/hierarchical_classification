import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout, Flatten


def conv_block(units, kernel_size=(3, 3), stride=(1, 1), dropout=0.3, activation='relu', block=1, layer=1, class_type='default', enable_dropout=True, initializer=tf.keras.initializers.GlorotNormal()):
    def layer_wrapper(inp):
        x = Conv2D(units, kernel_size, padding='same', kernel_initializer=initializer, name=f'{class_type}_block{block}_conv{layer}')(inp)
        x = Activation(activation, name=f'{class_type}_block{block}_act{layer}')(x)
        x = BatchNormalization(name=f'{class_type}_block{block}_bn{layer}')(x)
        if enable_dropout:
            x = Dropout(dropout, name=f'{class_type}_block{block}_dropout{layer}')(x)
        return x

    return layer_wrapper


def dense_block(units, dropout=0.3, activation='relu', name='default_fc1', enable_dropout=True, initializer=tf.keras.initializers.GlorotNormal()):
    def layer_wrapper(inp):
        x = Dense(units, kernel_initializer=initializer, name=name)(inp)
        x = Activation(activation, name='{}_act'.format(name))(x)
        if enable_dropout:
            x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x

    return layer_wrapper


def vgg11(img_input, conv_dropout=0.2, activation='relu', class_type='default', nodes=64):
    # Block 1
    block = 1
    x = conv_block(nodes, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(img_input)
    x = MaxPooling2D((2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 2
    block += 1
    x = conv_block(nodes * 2, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = MaxPooling2D((2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 3
    block += 1
    x = conv_block(nodes * 4, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 4, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 4
    block += 1
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 5
    block += 1
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), name=f'{class_type}_block{block}_pool')(x)

    x = Flatten(name=f'{class_type}_flatten')(x)

    return x


def vgg13(img_input, conv_dropout=0.2, activation='relu', class_type='default', nodes=64):
    # Block 1
    block = 1
    x = conv_block(nodes, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(img_input)
    x = conv_block(nodes, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 2
    block += 1
    x = conv_block(nodes * 2, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 2, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 3
    block += 1
    x = conv_block(nodes * 4, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 4, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 4
    block += 1
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=f'{class_type}_block{block}_pool')(x)

    # Block 5
    block += 1
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=1, class_type=class_type)(x)
    x = conv_block(nodes * 8, dropout=conv_dropout, activation=activation, block=block, layer=2, class_type=class_type, enable_dropout=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=f'{class_type}_block{block}_pool')(x)

    x = Flatten(name=f'{class_type}_flatten')(x)

    return x
