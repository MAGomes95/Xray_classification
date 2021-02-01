from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.xception import Xception
from tensorflow import reshape
from tensorflow import reduce_max
from tensorflow import reduce_mean
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SeparableConvolution2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import relu
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model


def learning_rate_schedule(
    initial_lr: float = 0.01
):
    """ Learning rate drop scheduler

    Args:
      initial_lr: Initial learning rate

    Returns:
      New learning rate
    """
    def exponential_decay(
      epoch: int
    ) -> float:

        return initial_lr * 0.98 ** epoch

    return exponential_decay


def generator_training(
        model,
        train_generator,
        validation_data,
        class_weights: dict,
        epochs: int,
        train_steps: int,
        callbacks: tuple = (
            ModelCheckpoint(
                filepath='../models/trained/Xception',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            LearningRateScheduler(
                learning_rate_schedule(0.0006),
                verbose=1

            )
        )
):
    """Performs train and simultaneously validation

    The functions aims to simultaneously train and validate
    the given model, based on the respected generators

    Args:
        model: Model to train
        train_generator: generator, that returns train tuple instances (input, target)
        validation_data: Validation data, in the form (inputs, target)
        class_weights: class weights in a dictionary format
        epochs: Number of epochs to train
        train_steps: Steps per epoch, on training
        callbacks: callback functions to apply during training

    Returns:
        History object when metrics and loss value over training and validation
    """

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        callbacks=list(callbacks),
        validation_data=validation_data,
        class_weight=class_weights,
    )

    return history


def extraction_network(inputs, reduction_ratio=16):
    """Auxiliary Network for Channel Attention Map

    This function aims to implement an auxiliary network,
    that will be used in the calculation of the channel
    attention map.

    The network is composed by two layer, in which the first
    equals the number of channels * 1/reduction_ratio, and the
    second equals the number of channels of the input

    Args:
        inputs: inputs, in which extraction will be performed
        reduction_ratio: parameter that controls number of parameters in first layer

    Returns:
        Extracted features from the input
    """
    first_layer = Dense(
        int(inputs.shape[1] / reduction_ratio),
        activation='relu'
    )(inputs)

    last_layer = Dense(
        inputs.shape[1],
        activation='linear'
    )(first_layer)

    return last_layer


def channel_attention(inputs):
    """Channel Attention Map calculation

    The function aims to implement a simple
    version of a Channel Attention, whose output
    is a tensor that will weight each channel of the
    input.

    Args:
        inputs: Input tensor, above which the channel
        map will be calculated

    Returns:
        Channel Attention map
    """

    max_features = GlobalMaxPool2D()(inputs)
    avg_features = GlobalAvgPool2D()(inputs)

    extracted_max = extraction_network(max_features)
    extracted_avg = extraction_network(avg_features)

    merge = Add()[extracted_avg, extracted_max]

    return reshape(sigmoid(merge), (-1, 1, 1, inputs.shape[3]))


def spatial_attention(inputs):
    """Spatial Attention Map calculation

    The function aims to implement a simple
    version of a Spatial Attention, whose output
    is a tensor that will output in a generalized manner,
    across all channels, each position.

    Args:
        inputs: Input tensor, above which the spatial
        map will be calculated

    Returns:
        Spatial Attention map
    """

    max_features = reduce_max(inputs, axis=3, keepdims=True)
    avg_features = reduce_mean(inputs, axis=3, keepdims=True)

    concat = Concatenate()[max_features, avg_features]

    spatial_weights = Conv2D(
        filters=1, kernel_size=7,
        padding='same', activation='sigmoid'
    )(concat)

    return spatial_weights


def get_xception():
    """Returns a Xception pretrained neural net.

        The function returns a partially pretrained Xception neural net.

        Returns:
            A modified Xception semi-pre-trained NN instance
    """

    model = Xception(
        include_top=False
    )

    model.trainable = False

    core_output = model.layers[45].output

    # Weighting Xception output via channel and spatial Attention

    channel_attention_map = channel_attention(core_output)
    channel_weighted = core_output * channel_attention_map
    spatial_attention_map = spatial_attention(channel_weighted)
    core_output = channel_weighted * spatial_attention_map

    for _ in range(5):

        output = relu(core_output)
        output = SeparableConvolution2D(
            728, (3, 3), padding='same',
            depthwise_regularizer=L2(0.2),
            pointwise_regularizer=L2(0.03)
        )(output)
        output = BatchNormalization()(output)
        output = Dropout(0.3)(output)

        output = relu(output)
        output = SeparableConvolution2D(
            728, (3, 3), padding='same',
            depthwise_regularizer=L2(0.2),
            pointwise_regularizer=L2(0.03)
        )(output)
        output = BatchNormalization()(output)
        output = Dropout(0.3)(output)

        core_output = Add()[output, core_output]

        # Output Weighting via Attention

        channel_attention_map = channel_attention(core_output)
        channel_weighted = core_output * channel_attention_map
        spatial_attention_map = spatial_attention(channel_weighted)
        core_output = channel_weighted * spatial_attention_map

    model_output = GlobalAvgPool2D()(core_output)

    model_output = Dense(1, activation='sigmoid')(model_output)

    model = Model(inputs=model.input, outputs=model_output)

    return model

