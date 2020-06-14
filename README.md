# FacialRecognitionTransferLearning
### This task is meant to demonstrate the use of Transfer Learning for Facial Recognition. I have used mobileNet pretrained model which is a very popular model for detection of 100+ objects. Here I have used to train it on images from http://www.anefian.com/research/face_reco.htm and my images. The model trains on 12 images of 4 subjects and tests on 3 images each of these 4 subjects.Lets go ahesd and see how the code works.

## **COLLECTION OF FACIAL DATA**
#### The First Task is to collect the image and make a dataset with it for this we use the following CV2 code hich collects 15 images particularly of the face using the HaarCascade model
![CV2 img 1](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-14%20155344.png)
![CV2 img 1](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-14%20155410.png)

#### The above collected data is then divided into two folders for train and test as in the folder facedata in this repository

## **TRAINING THE MODEL**
#### Here we try to see how the model works
### **IMPORT MODULES AND THE MODEL** 
#### In this part of the program, we import all the modules required and also load the weights of MobileNet model without incliuding the output layer as we need to change it as per our requirement according to the number of people to be recognized
![MODEL1](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/model1.png)

### **ADDING LAYERS AND COMPILING THE MODEL** 
#### For transfer learning its important to fix the weight of the layers as it should not change. It is done by making trainable value false for each layer in the MobileNet model. At the same time we need to add extra layers to train our model so 3 Dense layers are added on top of it ans then the output layer the last dense layer is having 4 output as we are training the model on  4 different people. Then we compile the both untrainable part and trainable part into one using Model module. then the details of the model is checked to see if the  layers have joined properly
![MODEL2](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/model2.png)

### **IMAGE PREPROCESSING FOR TRAINING** 
#### Its important to preprocess the images before training. For this we use ImageDataGenerator of keras module and we try to process both trainin gand testing dat and import it from the directory where the data is stored.
![MODEL3](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/model3.png)

### **TRAINING THE MODEL ON PREPROCESSED IMAGES** 
#### Here the program use RMSprop optimizer and categorical crossentrophy as loss function for training the model for 3 epoches. It also has earlystopping and modelcheckpoint to ensure that the model is saved after every epoches. 
![MODEL4](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/model4.png)

## **LOADING SAVED MODEL** 
#### In this part the saved model is loaded for further predictions. 
![LOAD1](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-06%20150115.png)


## **PREDICTING THE IMAGES** 
#### In this part the categories of the folders are picked and proper names are assigned as  per the different people. Then the random images are picked and predicted. It also uses cv2 to properly format the images and write the output on it and display it. 
![PREDICT1](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-06%20150057.png)
![PREDICT2](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-06%20145958.png)

## **SOME SAMPLE RESULTS**

![RESULT1](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/1.png)
![REUSULT2](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/2.png)

![RESULT3](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-06%20145539.png)
![RESULT4](https://github.com/vikashkr437/FacialRecognitionTransferLearning/blob/master/Sceenshot/Annotation%202020-06-06%20134838.png)

