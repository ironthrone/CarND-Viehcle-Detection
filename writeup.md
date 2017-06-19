
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[explorerHog]: ./output_images/explorer_hog.png
[explorerColor]: ./output_images/explorer_color.png
[explorerSpatial]: ./output_images/explorer_spatial.png
[recognized1]: ./output_images/recognized1.png
[recognized2]: ./output_images/recognized2.png
[recognized3]: ./output_images/recognized3.png
[recognized4]: ./output_images/recognized4.png
[recognized5]: .//output_images/recognized5.png
[recognized6]: .//output_images/recognized6.png

[heatmap1]: .//output_images/heapmap1.png
[heatmap2]: .//output_images/heapmap2.png
[heatmap3]: .//output_images/heapmap3.png
[heatmap4]: .//output_images/heapmap4.png
[heatmap5]: .//output_images/heapmap5.png
[heatmap6]: .//output_images/heapmap6.png

[labeled1]: .//output_images/labeled1.png
[labeled2]: .//output_images/labeled2.png
[labeled3]: .//output_images/labeled3.png
[labeled4]: .//output_images/labeled4.png
[labeled5]: .//output_images/labeled5.png
[labeled6]: .//output_images/labeled6.png

[last6]: ./output_images/hist_frame_boxed_heatmap.png
[currBox]: ./output_images/current_frame_boxed.png
[currLabeled]: ./output_images/current_frame_labeled.png

[headacheCar]: ./test_images/headache_car.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code is located in classifier.py get_hog_features(),extract_feature(),and test.py explore()
I select some car and not car images from dataset,and check there hog feature in different color  channel: 'RGB','HLS','Gray'.This is one of the comparision image
![][explorerHog]
From these images,i observed `R`,`G`,`B`,`Gray` and `L` channel all can clear seperate car and not car, and they are similiar,so i select `Gray` channel to get hog feature.
But later i add the H ,S channel hog feature , it all because of the error predict on the yellow lane line.I will mention how i solve it in the discusstion part
![][headacheCar]

####2. Explain how you settled on your final choice of HOG parameters.
I select orient = 15,pix_per_cell = 12,cell_pre_block = 2.Amonng these parameters i only tune the orient, and then the project meet specification.
First my orient is 9,after i make the pipeline work , i got many False Positive value.Some False Positive window just normal road ,so i choice to increase the orient ,thus i get more hog feature, and the outline of car will become more clearly,it will help filter these simple "road window"

And in order to speed up, i try to increase the pix_per_cell,after some experimention ,i select 12

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code is located in the bottom of classifier.py 
I use the  provided viehcle and non-viehcle dataset.In the training process,i select 1000 image from each class as the training data.Then i split these data to training and test data,test data is 30%.

After extract feature using extract_feature(), i create a GridSearchCV to search best SVC params.I tried `linear`,`rbf`kernel,after some experiment, the linear kernel is proved to be better,so i just set kernel is linear,so you can not find the rbf kernel. And i set C is (0.01,0.05,0.1)
In order to get a low false positive, i set scoring to `percision`. Now it is time to train and evaluate, the result is good,i got a 
97.4% test accuracy

I also use color and spatial feature.i explored color and spatial feature in different channel, the code is also located test.py explore().This is color compare image,i observe some images and find R,G,B,L channel can be use to seperate car and not car.So in classifier.py extract_features() , i extract rgb and l channel color feature
![][explorerColor]
The spatial comparision image is this.These feature can not tell how to seperate car and not car clearly,so I just select gray spatial feature,size=(16,16)
![][explorerSpatial]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is located in marker.py search_and_classify()
I slide the image twice ,and i did not search the whole image.First window = (96,64),x_range = [400,None].y_range=[400,550],stride = (24,16)
second window = (128,96).x_range = [400,None].y_range=[400,600],
stride=(36,24)

I tried window=(64,64),but there is many false positive window, and the car is always rectangle,so i set window  rectangle.And in order to recogonize far and near car,i add a (128,96) window .
The overlap is based on one third of side length of window

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is the recognized window image:
![][recognized1]
![][recognized2]
![][recognized3]
![][recognized4]
![][recognized5]
![][recognized6]

In order to achieve this result.I do adjustion from these methods:

* increase the orient to 12
* use RGB and  L channel color histgram feature
* Use GridSearchCV to train data
* add H,S hog feature


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_marked.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code is located in marker.py heat_and_thresh(), track_video.py class Tracker.track() 

For single image,I use heatmap to record sum detected window's impact ,then apply a threshold value to remove lower heat value,that means to remove false positive value

When processing video,i record last 10 frame's detected window ,and after finding  windows on new frame , i use history window and these new windows toghter to a heatmap filter  to get a smooth and relialy result.The threshold value = 10 / 2.

### Here are six frames and their corresponding heatmaps:

![][last6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![][currLabeled]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![][currBox]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most headache problem for me is that the classifier always predit the yellow lane line is car, i crop a little image to show this is "car"
![][headacheCar]
I tried different color ,spatial channel,different hog parameter ,except different hog channel.,but problem was not resolved .Until a slack friend notify me that i can try different channel on hog .
At the begining,i just select gray hog feature,because the aboved explored feature shows that R,G,B,Gray channel give same result,and the different between car and not car on these channel is obvious,i think gray si enough. After being notified ,i realized S and H channel can also differentiate car and not car,these feature maybe help classifier on different part. Then i add H,S channel hog feature, the problem is solved

I think if change to another video , the threshold value maybe fail, this value maybe  not general. And i do not track very large car ,like truck,i will fail if meeting a truck on road

There are many i can improve ,like 

* get a real time speed,
* integrate with Project 4,
* capture a custom video and apply to it
* calculate distance with these detected car... 
But i almost excess the Big Deadline ,so i just leave these up

To conclusion , this project is different from P2,P3. in those project we use deeplearning to solve ComputerVision problem,these model just accept all feature in a image,i do not need to select feature. But SVM is not good for model with above 10000 feature, i have to consider extract which feature . The process shows that If i did not select right feature,the svm will get a bad performance. So The `feature selection` is most import for this project