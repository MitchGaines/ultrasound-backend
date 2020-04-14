import tensorflow as tf
import numpy as np
import os.path

curr_path = os.path.dirname(__file__)


def predict_from_image(image):
    image = np.reshape(image, [-1, 310, 128, 1])  # reshape image to matrix
    model = tf.keras.models.load_model(os.path.join(curr_path, './models/four_fingers/both_datasets_convolutional.h5'))
    preds = model.predict(image)
    preds = preds.flatten().tolist()  # convert one row matrix to a json serializable list
    return preds
