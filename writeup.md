# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization/example_im.jpg "Example Image"
[image2]: ./visualization/normalized.jpg "Normalized Image"
[image3]: ./examples/random_noise.jpg "Random Noise"
[myimage0]: ../mydata/myimage0.png "Traffic Sign 1"
[myimage1]: ../mydata/myimage1.png "Traffic Sign 2"
[myimage2]: ../mydata/myimage2.png "Traffic Sign 3"
[myimage3]: ../mydata/myimage3.png "Traffic Sign 4"
[myimage4]: ../mydata/myimage5.png "Traffic Sign 5"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) (I have not committed this to github yet.)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples.
* The size of the validation set is 4410 samples.
* The size of test set is 12630 samples.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

I started by selecting a random image and displaying it along with its label. A random image can be seen here:

![Example Image][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a last step, I normalized the image data so all my images have zero mean and equal variance.

Here is an example of an original image and the normalized version of that image:

![Original Image][image1]
![Normalized Image][image2]

The difference between the original data set and the normalized data is that the normalized data has mean zero and equal variance while the original may not.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation Layer								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| Activation Layer								|
| Max pooling	    	| 2x2 stride,  outputs 5x5x16   				|
| Flatten Conversion	| Input = 5x5x16, Output = 400  				|
| Fully connected		| Input = 400, Output = 120						|
| RELU					| Activation Layer								|
| Dropout				| Delete 38% to reduce overfitting				|
| Fully connected		| Input = 120, Output = 84						|
| RELU					| Activation Layer								|
| Dropout				| Delete 38% to reduce overfitting				|
| Fully connected		| Input = 84, Output = 43						|
| RELU					| Activation Layer								|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a very normal learning rate of 0.001. I tried both 0.01 and 0.0001 and both yielded much worse results. I had also set my batch size and number of epochs above to 128 and 10 respectively. I grabbed these numbers from the lenet lab and they worked well enough for this pipeline that I decided not to change them.

I then ran my lenet function pipeline on my x batch of images. This gave me my logits that I will use to determine my accuracy. 

Next I calculated the cross entropy with the tf.nn.softmax_cross_entropy_with_logits and my logits calculated above.

I then was able to calculate my loss by using the tf.reduce_mean function on my cross entropy variable I just calculated.

Finally I was able to define my optimizer with the learning rate of 0.001 and apply that to my loss to minimize it. Using this pipeline over and over should increase my accuracy. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.946
* test set accuracy of 0.930

I started off my pipeline with a well known architecture. This architecture was lenet. I chose this because it was well known to me because of the lenet lab. Unfortunately, lenet on its own did not yield a good enough validation accuracy. I had to make some tweaks to my pipeline. 

I first started by trying to pre-process the images just a tad, but none of the small alterations made a significant enough difference. This is when I decided to play with the input and output sizes in the lenet structure. Everything I did here appeared to make my accuracy worse and not better. This is when I decided to look through my notes to possibly add more layers.

I soon decided to test out a dropout layer right before the last fully connected layer with a 50% dropout rate. This improved my accuracies immensely so I kept it. I then tried to add another one in multiple different locations in the pipeline and it seems that right before my first fully connected layer was best. Then I realized that my pipeline was overfitting just a bit so I made the dropout rate .62% in order to delete about 38%. This increased my validation accuracy my a fair amount. After achieving a little over 94% accuracy I felt that my pipeline was sufficient.

In a more real world setting I would definitely want to spend more time preprocessing to achieve an even higher accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][myimage0] ![Traffic Sign 2][myimage1] ![Traffic Sign 3][myimage2] 
![Traffic Sign 4][myimage3] ![Traffic Sign 5][myimage4]

The first image is a Stop Sign: I could see this potentially being a problem if the neural network is not familiar with stop signs where the white edges go all the way to the edge of the photo. Most likely this will not be a problem though.

The second image is a General Caution Sign: The general caution sign is unlikely to pose a problem to this neural network because the exclamation point is so easy to make out here. 

The third image is a Keep Right Sign: This sign seems pretty easy to decipher as the neural network will be able to tell that it is round, blue, and the arror is facing diagonally down and to the right.

The fourth image is a Yield Sign: This one could be slightly difficult since it is a similar shape to a few others, but the center of the sign is all white which should be a dead giveaway for the neural network.

The fifth image is a Dangerous Curve to the Right Sign: This one shares the same shape as many others and many of those were extremely blurry so I could see the neural network having trouble being able to guess what this one is.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| General Caution   	| General Caution 								|
| Keep Right			| Keep Right									|
| Yield Sign	      	| Yield Sign					 				|
| Curve Right			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This does not compare amazing to the test set which had a test accuracy of...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.96), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96           		| Stop sign   									| 
| .036              	| Bicycles Crossing 							|
| .0017					| Yield     									|
| .0013 				| Road Work 					 				|
| .000366   		    | Speed Limit (30km/h)      					|


For the second image, the model is relatively sure that this is a U-turn sign (probability of 0.99), and the image does contain a U-turn sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99           		| U-turn    									| 
| 4.19957451e-06    	| Traffic Signals 								|
| 3.37312736e-06		| Pedestrians									|
| 6.71851333e-12  		| Right of Way					 				|
| 1.22398781e-15		    | Road Work         						|

For the third image, the model is completely sure that this is a Yield sign (probability of 1.00), and the image does contain a Yield sign. I have not included the top five softmax probabilities because the model is almost certain this is a yield sign.

For the fourth image, the model is completely sure that this is a Bumpy Road sign (probability of 1.00), and the image does contain a bumpy road sign. I have not included the top five softmax probabilities because the model is almost certain this is a bumpy road sign.

For the fifth image, the model is relatively sure that this is a slippery road sign (probability of 0.97), and the image does not contain a slippery road sign. It is slightly worrisome that the model was so confident in a slippery road sign, but the second most probable sign was the correct right turn sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97           		| Slippery Road     							| 
| .024               	| Curve to Right 								|
| .0092					| End of No Passing								|
| 7.75220615e-05  		| No Passing					 				|
| 5.97955022e-07        | Ahead Only        							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


