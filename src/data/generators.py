import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input


def get_generator(
        data_id: pd.DataFrame,
        image_dir: str,
        id_column: str,
        label_column: str,
        image_size: tuple,
        images_per_batch: int = 32,
        preprocessing=preprocess_input
):
    """Get train generator

    The functions generates a training generator,
    which yields batches of image data after some
    preprocessing steps: data normalization and resize

    Args:
        data_id: Data which holds the images ids and respected target
        image_dir: Folder in which images are held
        id_column: Column which specifies images id
        label_column: Column which specifies label
        images_per_batch: Images per batch, for training
        image_size: input image dimension after resizing
        preprocessing: Preprocessing function

    Returns:
        Generator for training set
    """

    image_generator = ImageDataGenerator(
        preprocessing_function=preprocessing,
        dtype=tf.float32
    )

    data_id[label_column] = data_id[label_column].astype(str)

    generator = image_generator.flow_from_dataframe(
        dataframe=data_id,
        directory=image_dir,
        x_col=id_column,
        y_col=label_column,
        batch_size=images_per_batch,
        target_size=image_size,
        color_mode='rgb',
        class_mode='binary',
        shuffle=True
    )

    return generator

