import os
import cv2
import csv
import sklearn
import math
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten , Dense , Lambda , Activation, Cropping2D, Dropout
from tensorflow.keras.layers import Convolution2D
# from tensorflow.keras.layers import MaxPooling2D

_cnn_driving_log_file = [#'./train_data/CCW_data_3laps/driving_log.csv',
                        './train_data/CW_smooth_corner/driving_log.csv', 
                        './train_data/Recovery_left/driving_log.csv', 
                        # './train_data/Smooth_corner_turnright_1/driving_log.csv',
                        ]

_cnn_driving_folderpath = [#'./train_data/CCW_data_3laps/IMG/',
                        './train_data/CW_smooth_corner/IMG/', 
                        './train_data/Recovery_left/IMG/', 
                        # './train_data/Smooth_corner_turnright_1/IMG/',
                        ]

# _cnn_driving_log_file = ['./train_data/CCW_data_3laps/driving_log.csv','./train_data/CW_smooth_corner/driving_log.csv', './train_data/Recovery_data/driving_log.csv', './train_data/Recovery_data_new/driving_log.csv', './train_data/Recovery_data_corner/driving_log.csv', './train_data/Recovery_left/driving_log.csv', './train_data/Recovery_right/driving_log.csv']
# _cnn_driving_folderpath = ['./train_data/CCW_data_3laps/IMG/','./train_data/CW_smooth_corner/IMG/', './train_data/Recovery_data/IMG/', './train_data/Recovery_data_new/IMG/', './train_data/Recovery_data_corner/IMG/','./train_data/Recovery_left/IMG/', './train_data/Recovery_right/IMG/']

# _cnn_driving_log_file = ['./train_data/CCW_data/driving_log.csv', './train_data/CW_smooth_corner/driving_log.csv', './train_data/Recovery_data/driving_log.csv', './train_data/Recovery_data_new/driving_log.csv', './train_data/Recovery_data_corner/driving_log.csv', './train_data/Recovery_left/driving_log.csv', './train_data/Recovery_right/driving_log.csv']
# _cnn_driving_folderpath = ['./train_data/CCW_data/IMG/', './train_data/CW_smooth_corner/IMG/', './train_data/Recovery_data/IMG/', './train_data/Recovery_data_new/IMG/', './train_data/Recovery_data_corner/IMG/','./train_data/Recovery_left/IMG/', './train_data/Recovery_right/IMG/']

# _cnn_driving_log_file = [ './train_data/Recovery_left/driving_log.csv', './train_data/Recovery_right/driving_log.csv']
# _cnn_driving_folderpath = [ './train_data/Recovery_left/IMG/', './train_data/Recovery_right/IMG/']

_cnn_input_shape = (160,320,3)
_cnn_dropout_ratio = 0.5
_cnn_batch_size = 32
_cnn_learning_rate = 0.00095
_cnn_lines = []



# Models output folder
_cnn_model_output = "Models"

def _cnn_image_aug(img, steering_meas):
    img_flipped = np.fliplr(img)
    steering_flipped = -steering_meas
    return img_flipped, steering_flipped


if __name__ == '__main__': 

    ### read driving_log file
    for file, filepath in zip(_cnn_driving_log_file, _cnn_driving_folderpath):
        # print(file)
        # print(filepath)
        with open(file) as csvfile: 
            reader = csv.reader(csvfile)
            for line in reader: 
                img_path = filepath+line[0].split('/')[-1]
                print("img_path 0: ", img_path)
                line[0] = img_path
                img_path = filepath+line[1].split('/')[-1]
                print("img_path 1: ", img_path)
                line[1] = img_path
                img_path = filepath+line[2].split('/')[-1]
                print("img_path 2 ", img_path)
                line[2] = img_path
                # print(line)
                _cnn_lines.append(line)
                

    _cnn_train_samples, _cnn_validation_samples = train_test_split(_cnn_lines, test_size=0.2)
    print('lenf of input sample is', len(_cnn_lines))
    print('len of train_samples is', len(_cnn_train_samples))
    print('len of validation_samples is', len(_cnn_validation_samples))

    correction = 0.15 # this is a parameter to tune
    def _cnn_generator(samples, batch_size=32):
        num_samples = len(_cnn_lines)  
        while 1: # Loop forever so the generator never terminates
            shuffle(_cnn_lines)
            for offset in range(0, num_samples, batch_size):
                # print("offset:", offset)
                batch_samples = _cnn_lines[offset:offset+batch_size]
                _cnn_images = []
                _cnn_angles = [] 
                for batch_sample in batch_samples:
                    center_name = batch_sample[0]
                    left_name = batch_sample[1]
                    right_name = batch_sample[2]
                    # print("center name:", center_name)
                    ### read image
                    center_image = cv2.imread(center_name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2YUV)
                    left_image = cv2.imread(left_name)
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2YUV)
                    right_image = cv2.imread(right_name)
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2YUV)
                    ### read measured angle
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle + correction  
                    right_angle =    - correction
                    ### data augmentation
                    flipped_center_image, flipped_center_angle  = _cnn_image_aug(center_image, center_angle)
                    flipped_left_image, flipped_left_angle      = _cnn_image_aug(left_image, left_angle)
                    flipped_right_image, flipped_right_angle    = _cnn_image_aug(right_image, right_angle)            
                    ### append 
                    _cnn_images.append(center_image)
                    _cnn_images.append(left_image)
                    _cnn_images.append(right_image)
                    _cnn_images.append(flipped_center_image)
                    _cnn_images.append(flipped_left_image)
                    _cnn_images.append(flipped_right_image)            
                    _cnn_angles.append(center_angle)
                    _cnn_angles.append(left_angle)
                    _cnn_angles.append(right_angle)
                    _cnn_angles.append(flipped_center_angle)
                    _cnn_angles.append(flipped_left_angle)
                    _cnn_angles.append(flipped_right_angle)
   
                # trim image to only see section with road
                X_train = np.array(_cnn_images)
                y_train = np.array(_cnn_angles)
                print("len of X_train: {} and y_train: {}", len(X_train), len(y_train))
                yield sklearn.utils.shuffle(X_train, y_train)
            
    ### compile and train the model using the generator function
    _cnn_train_generator = _cnn_generator(_cnn_train_samples, batch_size=_cnn_batch_size)
    _cnn_validation_generator = _cnn_generator(_cnn_validation_samples, batch_size=_cnn_batch_size)
    ### build NN model based on Nvidia Architecture
    model = tf.keras.models.Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0 ,input_shape= (160,320,3)))
    model.add(Cropping2D(cropping = ((50,20) ,(0,0))))
    model.add(Convolution2D(24,5,5, subsample = (2,2),activation = 'relu'))
    model.add(Convolution2D(36,5,5, subsample = (2,2),activation = 'relu'))
    model.add(Convolution2D(46,5,5,subsample = (2,2) ,activation ='relu'))
    model.add(Convolution2D(64,3,3, activation = 'relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(1))
    
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

    history_object = model.fit(_cnn_train_generator, 
            steps_per_epoch=math.ceil(len(_cnn_train_samples)/_cnn_batch_size), 
            validation_data=_cnn_validation_generator, 
            validation_steps=math.ceil(len(_cnn_validation_samples)/_cnn_batch_size), 
            callbacks=_cnn_callbacks,
            epochs=5, verbose=1)
    
    # history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

    model.save('Models/model.h5')

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig('training_loss_' + date_string + '.png')
