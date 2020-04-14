import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error

matplotlib.use('Agg')

curr_path = os.path.dirname(__file__)


def test_cnn(data_type):
    test_x = []
    test_y = []
    title = ""
    if data_type == 'open_close_test':
        test_x = np.load(
            os.path.join(curr_path, '../data_dependencies/labeled_data/fist_relax/four_fingers/us_test.npy')).flatten()
        test_y = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/fist_relax/four_fingers/ang_test'
                                                 '.npy'))
        title = 'CNN: Open and Close Test Data:'
    elif data_type == 'open_close_train':
        test_x = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/fist_relax/four_fingers/us_train'
                                                 '.npy')).flatten()
        test_y = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/fist_relax/four_fingers/ang_train'
                                                 '.npy'))
        title = 'CNN: Open and Close Train Data:'
    elif data_type == 'pinch_relax_test':
        test_x = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/pinch_relax/four_fingers/us_test'
                                                 '.npy')).flatten()
        test_y = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/pinch_relax/four_fingers/ang_test'
                                                 '.npy'))
        title = 'CNN: Pinch and Relax Test Data:'
    elif data_type == 'pinch_relax_train':
        test_x = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/pinch_relax/four_fingers/us_train'
                                                 '.npy')).flatten()
        test_y = np.load(os.path.join(curr_path, '../data_dependencies/labeled_data/pinch_relax/four_fingers'
                                                 '/ang_train.npy'))
        title = 'CNN: Pinch and Relax Train Data:'

    test_x = np.reshape(test_x, [-1, 310, 128, 1])

    model = tf.keras.models.load_model(os.path.join(curr_path, './models/open_close_convolutional.h5'))
    # model = tf.keras.models.load_model(os.path.join(curr_path, './models/both_datasets_convolutional.h5'))
    # model = tf.keras.models.load_model(os.path.join(curr_path, './models/pinch_relax_convolutional.h5'))

    print('predictions...')
    preds = model.predict(test_x)

    # address ending error:
    for i in range(len(preds)):
        if preds[i] < 0 or preds[i] > 2:
            preds[i] = 0.25

    plt.clf()
    plt.plot(test_y, 'b')
    plt.plot(preds, 'r')
    plt.savefig(os.path.join(curr_path, '../static/cnn_performance'))

    cnn_mse = mean_squared_error(test_y, preds)
    score = 'CNN Mean Squared Error: ' + str(cnn_mse)

    return score, title
