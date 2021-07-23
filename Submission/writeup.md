# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


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

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_release.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ./Models/model_release.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 & 5x5 filter sizes and depths between 24 and 64 (model.py lines 166-204) 

The model includes RELU layers to introduce nonlinearity (code line 179), and the data is normalized in the model using a Keras lambda layer (code line 174). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 180). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13-48 and line 227). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 218).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination listed below:

    * 2x center lane driving, clockwise
    * 5x recovery from the side 
    * 7x smooth corner left and right turns

For details about how I created the training data, see the next section. 

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

I used an adam optimizer so that manually training the learning rate wasn't necessary.