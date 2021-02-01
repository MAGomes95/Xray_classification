import pandas as pd


def class_weights(
        target_column: pd.Series
) -> dict:
    """Calculate weight of each class in training

    The weights are calculated through the count of
    each instance in each class, in order to, weight
    the loss function

    Args:
        target_column: Column containing the target

    Returns:
        Weight for each class
    """
    target_column = target_column.astype(int)

    frequencies = target_column.value_counts() / target_column.shape[0]

    output = {
        0: float(frequencies.loc[frequencies.index == 1]),
        1: float(frequencies.loc[frequencies.index == 0])
    }

    return output
