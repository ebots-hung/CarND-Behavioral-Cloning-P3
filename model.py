import os
import cv2
import csv
import sklearn
import math
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

_cnn_driving_log_file = './Beh_cloning_Sim_data/driving_log.csv'
_cnn_input_shape = (160,320,3)
_cnn_dropout_ratio = 0.5
_cnn_batch_size = 32
_cnn_learning_rate = 0.00095
_cnn_lines = []

_cnn_center_img_path = []
_cnn_left_img_path = []
_cnn_right_img_path = []
_cnn_steer_angle_meas = []
_cnn_throttle_meas = []
_cnn_break_meas = []
_cnn_speed_meas = []

# Models output folder
_cnn_model_output = "Models"

def _cnn_image_aug(img, steering_meas):
    img_flipped = np.fliplr(img)
    steering_flipped = -steering_meas
    return img_flipped, steering_flipped


if __name__ == '__main__': 

    ### read driving_log file
    
    with open(_cnn_driving_log_file) as csvfile: 
        reader = csv.reader(csvfile)
        for line in reader: 
            _cnn_lines.append(line)

    _cnn_train_samples, _cnn_validation_samples = train_test_split(_cnn_lines, test_size=0.2)
    # print('lenf of input sample is', len(_cnn_lines))
    # print('len of train_samples is', len(_cnn_train_samples))
    # print('len of validation_samples is', len(_cnn_validation_samples))

    # print(_cnn_lines[0])
    ### read images
    # for cnnline in _cnn_lines: 
    #     _cnn_center_img_path.append(cnnline[0])
    #     _cnn_left_img_path.append(cnnline[1])
    #     _cnn_right_img_path.append(cnnline[2])
    #     _cnn_steer_angle_meas.append(cnnline[3])
    #     _cnn_throttle_meas.append(cnnline[4])
    #     _cnn_break_meas.append(cnnline[5])
    #     _cnn_speed_meas.append(cnnline[6])

    

    # print(_cnn_center_img_path[0])
    # print(_cnn_left_img_path[0])
    # print(_cnn_right_img_path[0])
    # print(_cnn_steer_angle_meas[0])
    # print(_cnn_throttle_meas[0])
    # print(_cnn_break_meas[0])
    # print(_cnn_speed_meas[0])
    batch_size=32
    # def _cnn_generator(samples, batch_size=32):
    num_samples = len(_cnn_lines)
    # while 1: # Loop forever so the generator never terminates
    shuffle(_cnn_lines)
    for offset in range(0, num_samples, batch_size):
        batch_samples = _cnn_lines[offset:offset+batch_size]
        images = []
        angles = []
        for batch_sample in batch_samples:
            name = './Beh_cloning_Sim_data/IMG/'+batch_sample[0].split('/')[-1]
            center_image = cv2.imread(name)
            center_angle = float(batch_sample[3])
            images.append(center_image)
            angles.append(center_angle)
        # trim image to only see section with road
        X_train = np.array(images)
        y_train = np.array(angles)
        # yield sklearn.utils.shuffle(X_train, y_train)
            
    ### compile and train the model using the generator function
    # _cnn_train_generator = _cnn_generator(_cnn_train_samples, batch_size=_cnn_batch_size)
    # _cnn_validation_generator = _cnn_generator(_cnn_validation_samples, batch_size=_cnn_batch_size)
    ### build NN model based on Nvidia Architecture
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Lambda(lambda x: x/127.5 - 1., input_shape=_cnn_input_shape, output_shape=_cnn_input_shape))

    model.add(tf.keras.layers.Cropping2D(input_shape=_cnn_input_shape, 
                     cropping=((70,25),(0,0))))

    model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(rate=_cnn_dropout_ratio))

    model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(rate=_cnn_dropout_ratio))

    model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=5, strides=(2, 2),activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(rate=_cnn_dropout_ratio))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(rate=_cnn_dropout_ratio))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),activation='relu', padding='valid'))
    model.add(tf.keras.layers.Dropout(rate=_cnn_dropout_ratio))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1164, activation='relu'))

    model.add(tf.keras.layers.Dense(100, activation='relu'))

    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dense(1))

    model.summary()

    ### save model output
    if os.path.exists(_cnn_model_output):
        print("Models output folder existed!!!")
    else: 
        os.mkdir(_cnn_model_output)
        print("Models output folder created at: {}".format(os.path.join(os.getcwd(),_cnn_model_output)))

    _cnn_callbacks = (
        tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=3),
        tf.keras.callbacks.ModelCheckpoint(filepath='Models/model.{epoch:02d}-{val_loss:.4f}.h5'),
    )

    model.compile(loss='mse', optimizer='adam')

    # model.fit_generator(_cnn_train_generator, 
    #         steps_per_epoch=math.ceil(len(_cnn_train_samples)/_cnn_batch_size), 
    #         validation_data=_cnn_validation_generator, 
    #         validation_steps=math.ceil(len(_cnn_validation_samples)/_cnn_batch_size), 
    #         callbacks=_cnn_callbacks,
    #         epochs=5, verbose=1)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20)

    model.save('Models/model.h5')