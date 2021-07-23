# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model_release.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

[//]: # (Image References)

[image1a]: ./output_images/trained_image_-0.1971831_bgr.png "Data Visualization"
[image1b]: ./output_images/trained_image_-0.1971831_gray.png "Data Grayscaling"
[image1c]: ./output_images/trained_image_-0.1971831_yuv.png "Data YUV"

[image2a]: ./output_images/angle_distribution_no_augmentation2021_07_20_18_26_50.png "Data distribution before augmentation"
[image2b]: ./output_images/angle_distribution2021_07_20_18_22_39.png "Data distribution after augmentation"

[image3a]: ./output_images/recovery_left_img_bgr.png "Recovery Image From The Left"
[image3b]: ./output_images/recovery_right_img_bgr.png "Recovery Image From The Right"

[image4a]: ./output_images/nocrop_image_bgr.png "Normal Image"
[image4b]: ./output_images/cropped_img_50_20_bgr.png "Cropped Image (50,20)(0,0)"
[image4c]: ./output_images/cropped_img_70_25_bgr.png "Cropped Image (70,25)(0,0)"

[image5a]: ./output_images/training_loss_epochs_5.png "Training loss - 5 epoches"
[image5b]: ./output_images/training_loss_epochs_7.png "Training loss - 7 epoches"
[image5c]: ./output_images/training_loss_epochs_20.png "Training loss - 20 epoches"
[image5d]: ./output_images/training_loss_final_model.png "Training loss - final model - 7 epoches"

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `./Models/model_release.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py ./Models/model_release.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

### How to run Udacity Car-sim with Nvidia GPU
    RUN_GRAPH=true /home/hunglam/learning/CarND/term1-simulator-linux/beta_simulator_linux/beta_simulator.x86_64

### Troubleshotting reference
    https://github.com/udacity/self-driving-car-sim/issues/131

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find existing model with similar application.

My first step was to use a convolution neural network model similar to NVIDIA Self-Driving Car model. I thought this model might be appropriate because it trains the model by using multiple camera setup & also output the steering angles.

To avoid the overfitting, I modified the model so that I tried different dropout ratio. Also try with different number of epochs, cropping area. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I played with different corrections in left/right images and adding smooth corner data for training.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 166-204) consisted of a convolution neural network with the following layers and layer sizes _________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d (Cropping2D)      (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d (Conv2D)              (None, 31, 158, 24)       1824      
_________________________________________________________________
dropout (Dropout)            (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_4 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten (Flatten)            (None, 2112)              0         
_________________________________________________________________
dense (Dense)                (None, 1164)              2459532   
_________________________________________________________________
dense_1 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 51        
=================================================================
Total params: 2,712,481
Trainable params: 2,712,481
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4a]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn

![alt text][image3a]
![alt text][image3b]

To augment the data sat, I also flipped images and angles thinking that this would have symmetric data distribution on positive & negative steering angle.

![alt text][image2a]
![alt text][image2b]

Data is normalized by Lambda function and before feed into the training, I also cropped out the unecessary area such as sky, so that the NN would focus on the lane line. I tried with 2 different cropping area (50,20)(0,0) and (70,25)(0,0):
![alt text][image4b]
![alt text][image4c]

After the collection process, I had 15433 number of data points. I tried with different color space BGR, Grayscale and YUV, however Grayscale could not give good training output, then I finally stay with normal input from BGR. 
![alt text][image1a]
![alt text][image1b]
![alt text][image1c]

I finally randomly shuffled the data set and put 15% of the data into a validation set. (I use model.fit() with validation_slit 15%, batch_size = 32)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7

![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]    