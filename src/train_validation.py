from src.utils.identifiers import generate_id_dataframe
from src.utils.weights import class_weights
from src.models.train_validation import get_xception
from src.data.generators import get_generator
from src.models.train_validation import generator_training
from src.features.preprocessing import preprocessing
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    # Get and compile the model

    Xception = get_xception()

    Xception.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Identification dataframe retrieval

    train_localization = generate_id_dataframe(
        path_to_data='../data/bronze/train',
        saving_path='../data/silver/train_id.csv'
    )
    validation_localization = generate_id_dataframe(
        path_to_data='../data/bronze/val',
        saving_path='../data/silver/validation_id.csv'
    )
    test_localization = generate_id_dataframe(
        path_to_data='../data/bronze/test',
        saving_path='../data/silver/testing_id.csv'
    )

    # Class Weights calculation

    weights = class_weights(
        target_column=train_localization['label']
    )

    # Get training generator and validation set ready for Xception

    training_generator = get_generator(
        data_id=train_localization,
        image_dir='../data/bronze/train/BOTH/',
        id_column='Image',
        label_column='label',
        image_size=(299, 299)
    )
    validation_data = ()

    for row in validation_localization.index:

        row_image = validation_localization.loc[row, 'Image']
        row_target = validation_localization.loc[row, 'label']

        validation_data = preprocessing(
            image_path=f'../data/bronze/val/validation_xray/{row_image}',
            image_target=row_target,
            container=validation_data
        )

    # Training and Validation of Xception

    steps_train = training_generator.n // training_generator.batch_size

    Xception_history = generator_training(
        model=Xception,
        train_generator=training_generator,
        validation_data=validation_data,
        class_weights=weights,
        epochs=120,
        train_steps=steps_train
    )

    # Evaluation of the model in the Test data

    test_data = ()

    for row in test_localization.index:

        row_image = test_localization.loc[row, 'Image']
        row_target = test_localization.loc[row, 'label']

        test_data = preprocessing(
            image_path=f'../data/bronze/test/test_xray/{row_image}',
            image_target=row_target,
            container=test_data
        )

    inputs, targets = test_data[0], test_data[1]

    model = load_model('../models/trained/Xception')

    model.evaluate(inputs, targets)

