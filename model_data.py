import os
import cv2
import csv
import sklearn
import math
import datetime
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

_cnn_driving_log_file = [
    './train_data/CCW_data_1lap_1/driving_log.csv',  
    './train_data/CCW_data_1lap_2/driving_log.csv',   
    './train_data/Recovery_left_1/driving_log.csv', 
    './train_data/Recovery_left_2/driving_log.csv', 
    './train_data/Recovery_right_1/driving_log.csv',
    './train_data/Recovery_right_2/driving_log.csv',
    './train_data/Recovery_right_3/driving_log.csv',
    './train_data/Smooth_corner_leftturn_1/driving_log.csv', 
    './train_data/Smooth_corner_leftturn_2/driving_log.csv',
    './train_data/Smooth_corner_leftturn_3/driving_log.csv', 
    './train_data/Smooth_corner_leftturn_4/driving_log.csv',
    './train_data/Smooth_corner_leftturn_5/driving_log.csv',
    # './train_data/Smooth_corner_leftturn_6/driving_log.csv',  
    # './train_data/Smooth_corner_leftturn_7/driving_log.csv',       
    './train_data/Smooth_corner_turnright_1/driving_log.csv', 
    './train_data/Smooth_corner_turnright_2/driving_log.csv',
    ]
    
_cnn_driving_folderpath = [
    './train_data/CCW_data_1lap_1/IMG/',
    './train_data/CCW_data_1lap_2/IMG/',
    './train_data/Recovery_left_1/IMG/', 
    './train_data/Recovery_left_2/IMG/', 
    './train_data/Recovery_right_1/IMG/', 
    './train_data/Recovery_right_2/IMG/', 
    './train_data/Recovery_right_3/IMG/',     
    './train_data/Smooth_corner_leftturn_1/IMG/', 
    './train_data/Smooth_corner_leftturn_2/IMG/',
    './train_data/Smooth_corner_leftturn_3/IMG/',   
    './train_data/Smooth_corner_leftturn_4/IMG/',
    './train_data/Smooth_corner_leftturn_5/IMG/',
    # './train_data/Smooth_corner_leftturn_6/IMG/',   
    # './train_data/Smooth_corner_leftturn_7/IMG/',  
    './train_data/Smooth_corner_turnright_1/IMG/', 
    './train_data/Smooth_corner_turnright_2/IMG/',
    ]

_cnn_input_shape = (160,320,3)
_cnn_lines = []
_cnn_nb_lines = []

_cnn_images = []
_cnn_angles = [] 
# Models output folder
_cnn_model_output = "Models"

def _cnn_image_aug(img, steering_meas):
    img_flipped = np.fliplr(img)
    steering_flipped = -steering_meas
    return img_flipped, steering_flipped

def exploreData(savename, images, labels):
    imgLength = len(images)
    assert(imgLength == len(labels))  
    
    index = random.randint(0, imgLength)
    print(index)
    image = images[index]
    lable = labels[index]
    print("lable:", lable)
    filepath = os.path.join(savename + '_' + str(lable) + '_bgr.png')
    print(filepath)
    cv2.imwrite(filepath, image)
    filepath = os.path.join(savename + '_' + str(lable) + '_yuv.png')
    print(filepath)
    cv2.imwrite(filepath, cv2.cvtColor(image,cv2.COLOR_BGR2YUV))
    
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    filepath = os.path.join(savename + '_' + str(lable) + '_gray.png')
    print(filepath)
    cv2.imwrite(filepath, img_gray)
    
    img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    filepath = os.path.join(savename + '_' + str(lable) + '_rgb.png')
    print(filepath)

    cv2.imwrite(filepath, img_rgb)
    print("brg shape:",image.shape)
    print("grayscale shape:",img_gray.shape)
    print("img_rgb shape:",img_rgb.shape)

if __name__ == '__main__': 

    ### read driving_log file
    for file in _cnn_driving_log_file:
        # print(file)
        with open(file) as csvfile: 
            reader = csv.reader(csvfile)
            _cnn_nb_lines.append(len(list(reader)))

    for file in _cnn_driving_log_file:
        # print(file)
        with open(file) as csvfile: 
            reader = csv.reader(csvfile)
            for line in reader: 
                _cnn_lines.append(line)
                # print(line)

    print(len(_cnn_lines))
    print(_cnn_nb_lines)
    # _cnn_train_samples, _cnn_validation_samples = train_test_split(_cnn_lines, test_size=0.2)
    # print('lenf of input sample is', len(_cnn_lines))
    # print('len of train_samples is', len(_cnn_train_samples))
    # print('len of validation_samples is', len(_cnn_validation_samples))

    # batch_size = 32
    correction = 0.2 # this is a parameter to tune
    # def _cnn_generator(samples, batch_size=32):
    num_samples = len(_cnn_lines)  
    # while 1: # Loop forever so the generator never terminates
    shuffle(_cnn_lines)
    # for offset in range(0, num_samples, batch_size):
    offset = 0 
    for size, folderpath in zip(_cnn_nb_lines,_cnn_driving_folderpath): 
        # print(size)
        # print(folderpath)    
        # print(offset)    
        batch_samples = _cnn_lines[offset:offset+size]
        for batch_sample in batch_samples:
            center_name = folderpath+batch_sample[0].split('/')[-1]
            left_name = folderpath+batch_sample[1].split('/')[-1]
            right_name = folderpath+batch_sample[2].split('/')[-1]
            # print(center_name)
            ### read image
            center_image = cv2.imread(center_name)
            # center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2YUV)
            left_image = cv2.imread(left_name)
            # left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2YUV)
            right_image = cv2.imread(right_name)
            # right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2YUV)
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
            # _cnn_images.append(flipped_center_image)
            # _cnn_images.append(flipped_left_image)
            # _cnn_images.append(flipped_right_image)            
            _cnn_angles.append(center_angle)
            _cnn_angles.append(left_angle)
            _cnn_angles.append(right_angle)
            # _cnn_angles.append(flipped_center_angle)
            # _cnn_angles.append(flipped_left_angle)
            # _cnn_angles.append(flipped_right_angle)
        ### update offset
        offset = offset + size

    # trim image to only see section with road
    X_train = np.array(_cnn_images)
    y_train = np.array(_cnn_angles)
    print("len of X_train: {} and y_train: {}", len(X_train), len(y_train))
        # yield sklearn.utils.shuffle(X_train, y_train)
            
    # 'Plot X_data'
    img_path = os.getcwd()+'/'+'trained_image'
    exploreData(img_path, X_train, y_train)

    # 'Plot right corner image'
    img_file = os.getcwd() + '/train_data/Smooth_corner_leftturn_1/IMG/' + 'center_2021_07_20_00_27_30_739.jpg'
    # print(img_file)
    plt_img = cv2.imread(img_file)
    # print(plt_img.shape)
    save_file = 'smooth_corner_leftturn_img'
    cv2.imwrite(save_file+'_bgr.png', plt_img)
    cv2.imwrite(save_file+'_rgb.png', cv2.cvtColor(plt_img, cv2.COLOR_BGR2RGB))

    filepath = os.path.join(save_file + '_yuv.png')
    # print(filepath)
    cv2.imwrite(filepath, cv2.cvtColor(plt_img,cv2.COLOR_BGR2YUV))
    
    filepath = os.path.join(save_file + '_gray.png')      
    img_gray = cv2.cvtColor(plt_img,cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(filepath, img_rgb)

    ### 'Plot recovery left/right image'
    ### left
    img_file = os.getcwd() + '/train_data/Recovery_left_1/IMG/' + 'center_2021_07_17_16_33_41_832.jpg'
    # print(img_file)
    plt_img = cv2.imread(img_file)
    # print(plt_img.shape)
    save_file = 'recovery_left_img'
    cv2.imwrite(save_file+'_bgr.png', plt_img)
    cv2.imwrite(save_file+'_rgb.png', cv2.cvtColor(plt_img, cv2.COLOR_BGR2RGB))

    filepath = os.path.join(save_file + '_yuv.png')
    # print(filepath)
    cv2.imwrite(filepath, cv2.cvtColor(plt_img,cv2.COLOR_BGR2YUV))
    
    filepath = os.path.join(save_file + '_gray.png')      
    img_gray = cv2.cvtColor(plt_img,cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(filepath, img_rgb)

    ### right
    img_file = os.getcwd() + '/train_data/Recovery_right_1/IMG/' + 'center_2021_07_17_16_31_08_770.jpg'
    # print(img_file)
    plt_img = cv2.imread(img_file)
    # print(plt_img.shape)
    save_file = 'recovery_right_img'
    cv2.imwrite(save_file+'_bgr.png', plt_img)
    cv2.imwrite(save_file+'_rgb.png', cv2.cvtColor(plt_img, cv2.COLOR_BGR2RGB))

    filepath = os.path.join(save_file + '_yuv.png')
    # print(filepath)
    cv2.imwrite(filepath, cv2.cvtColor(plt_img,cv2.COLOR_BGR2YUV))
    
    filepath = os.path.join(save_file + '_gray.png')      
    img_gray = cv2.cvtColor(plt_img,cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(filepath, img_rgb)


    ### 'Plot cropping image'
    img_file = os.getcwd() + '/train_data/Recovery_left_1/IMG/' + 'center_2021_07_17_16_33_41_832.jpg'
    # print(img_file)
    plt_img = cv2.imread(img_file)
    top_crop = 50
    bot_crop = 20
    crop_img = plt_img[top_crop:160-bot_crop, 0:320]
    save_file = 'cropped_img_50_20'
    cv2.imwrite(save_file+'_bgr.png', crop_img)
    cv2.imwrite(save_file+'_rgb.png', cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

    # print(filepath)    
    # 'Angles Distribution'
    num_bins = 20
    bins = (np.arange(num_bins+2)-(num_bins+1)/2)/10
    x_label = (np.arange(num_bins+1)-num_bins/2)/10
    num_samples_bin, _, _ = plt.hist(y_train, bins=bins , rwidth=0.5)
    # plt.plot(history_object.history['loss'])
    plt.plot(x_label, num_samples_bin)
    plt.title('Angle distribution')
    plt.ylabel('Number of samples')
    plt.xlabel('angles')
    date_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig('angle_distribution_no_augmentation' + date_string + '.png')
