# **Behavioral Cloning** 

## Project Writeup


---

**Behavioral Cloning Project**

The goal of this project was to train network to steer the car in a computer game; using a realistic game to record training sets is good practice.

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report (the present project writeup)


[//]: # (Image References)

[image1]: ./output/model_architecture.jpg "Model Visualization"
[image2]: ./output/center_driving.jpg "Driving"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb, a jupiter notebook containing the script to create and train the model
* drive.py, for driving the car in autonomous mode
* model.h5, containing a trained convolution neural network 
* writeup.md, summarizing the results
* drive.mp4, a movie at 60fps with the car driving in autonomous mode
* keras-gpu.yml, the environment used for training

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
**python drive.py model.h5**

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers:
* a cropping layer used to crop the not needed top and bottom from the image
* a lambda layer used to resize and normalize images
* three 5x5 convolution layers with double stride and 'relu' activation
* two 3x3 convolution layers with normal stride and 'relu' activation
* a flatten layer to put all values in a vector
* three fully connected layers
* the output with just a value, that would be the predicted steering angle 

#### 2. Attempts to reduce overfitting in the model

No dropout was used on the model as in the development stage it was observed that it affected accuracy.

In order to avoid overfitting, the model was trained on data collected driving in clockwise direction, ensuring the test will be run on completely new images.
Second, to avoid overfitting, the recorded data was splitted in  trained and validated data sets.
Third, from the recorded, only the left and right images were used, as most of the center images had null steering angle; to ensure the network learns that for any image an action is expected, we ignored the center images; the steering angle was corrected before submiting to the network.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track; the car had no error on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
The angle correction was tunned to ensure the steering is decisive enough: at small values of corr, the car was steering slow and at sharp corners was getting out of the track.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. As said before, only lateral images were used to ensure the car learns to stay on the track; these imagaes were augmented with their mirrors; this augmentation actually helped as the model had many examples in wich it had to correct the steering.

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the knowledge I gatherrd in the course and come to a first solution; afterwards build up on it and improve the solution. I first used an architecture close to LeNet but it was not powerful enough; with this occasion I realized the difference between classification and regression (at first my care was outputing only one steering angle).

After that I start changing the model to what was actually very close to the one in the Nvidia paper. This is the model that I refined and I am going to explain in more detail in the following lines.

My first step was to create the generator; even if I have a realy powerful computer, I wanted to understand how generators work; the conclusion is this trick is usefull for large datasets, but it take a much longer time to train a network that storing all the data in the memory.

The second step was to create the model.
The first layer is cropping the images to keep only the relevant data; it's improtant to do this in the model, as it has to be done also on the test, when the car will actualy drive by itself.
The second layer uses the power of lambda functions to resize the images; this is of crucical importance as it affects the final size of the model; without resizing the model was 3 times larger.
Form this point on I used the Nvidia model: three 5x5 convolution layers with double stride and 'relu' activation, two 3x3 convolution layers with normal stride and 'relu' activation, a flatten layer to put all values in a vector, three fully connected layers, the output with just a value, that would be the predicted steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The errors were very close between the training set and validation set at each epoch, which suggested that the model was not overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and I improved that by increasing the correction for steering angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network. 
Here is a visualization of the architecture, with the layers and layers sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded **ten** laps on track one driving clockwise, using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would will ensure a more uniform distribution of steering decisions; this would also ensure there is a balance in steering to left and to right.

After the collection process, I had 9864 number of data points.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the stagnating loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
