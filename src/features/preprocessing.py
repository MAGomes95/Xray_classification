import numpy as np
from PIL.Image import open
from PIL.Image import BILINEAR
from tensorflow.keras.applications.xception import preprocess_input


def preprocessing(
        image_path: str,
        image_target: int,
        container: tuple,
        preprocessing_function=preprocess_input
) -> tuple:
    """Preprocesses a image instance and adds it to a container

   The given function aims to take a un-preprocessed image and its target,
    and add an tuple of the preprocessed image and the target array to the
    existing container

    Args:
      image_path: Path to the un-preprocessed image
      image_target: 1, 0 depending of the image class
      container: container of already preprocessed instances
      preprocessing_function:

    Returns:
      Updated container with the newly preprocessed instance and target

    """
    image = open(image_path).convert('RGB').resize((299, 299), BILINEAR)
    image_array = np.asarray(image).reshape((1, 299, 299, 3))
    target_array = np.asarray([image_target]).reshape((1, 1))

    preprocessed_image = preprocessing_function(image_array)

    if len(container) == 0:

        container = (preprocessed_image, target_array)

    else:

        container = (
            np.concatenate((container[0], preprocessed_image), axis=0),
            np.concatenate((container[1], target_array), axis=0)
        )

    return container

