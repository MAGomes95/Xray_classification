import os
import pandas as pd


def generate_id_dataframe(
        path_to_data: str,
        saving_path: str
) -> pd.DataFrame:
    """Generates and saves an identification dataframe to the images

    The function aims to generate and save a dataframe, in this case,
    with two columns:

    - A identifier of the image, i.e, the title
    - The label associated, being 1 normal and 0 pneumonia

    Args:
        path_to_data: Path to the folder where the images are
        saving_path: Path, where the csv will be stored

    Returns:
        A identification dataframe
    """

    normal_images = os.listdir(
        f'{path_to_data}/NORMAL/'
    )
    pneumo_images = os.listdir(
        f'{path_to_data}/PNEUMONIA/'
    )

    images = normal_images + pneumo_images
    images_label = [1]*len(normal_images) + [0]*len(pneumo_images)

    output = pd.DataFrame()
    output['Image'] = images
    output['label'] = images_label

    output['label'] = output['label'].astype(str)

    output.to_csv(saving_path)

    return output
