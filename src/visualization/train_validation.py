import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History


def training_plotting(
        history: History
) -> None:
    """Plotting training

    The function aims to produce a plot that describes
    the training process of a given model, plotting for
    that matter, the training loss and validation loss
    as a function of the number of epochs

    Args:
        history: History objected, returned by a fit method

    Returns:
        None
    """

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()

