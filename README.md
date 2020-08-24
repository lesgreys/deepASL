# The ASL Project
Applying machine learning techniques and convolutional neural nets to classify images of the American Sign Language (ASL) alphabet. This project contains several phases with long-term goals for real-world deployment. The intent is to have each subsequent phase build upon the previous phase. 

![helloASL](images/helloasl2.jpg)

**TL;DR**


## Why ASL

As a CODA, Child of Deaf Adult, for the record I have never used that term before until now, I have an intimate understanding of the communication problems between the deaf community and the rest of population. 

ASL has continously evolved into a more expressve and inclusive language but still bears extreme limitations when trying to communicate with those outside of the language. The deaf community constantly becomes siloed within their own groups and naturally have limitations in handling simple day to day task i.e. calling their phone provider to dispute or ask questions on their bill, call a doctor for an appointment, calling any customer service line, and countless other scenarios that I have personally experienced. A task that would take a hearing person 20-30 minutes can take a deaf person 2-3X longer, if they are not hung-up on because the business believes it's fraud. 

There are many issues in todays system of communication between the deaf and non-deaf, my goal is remove some of those barriers by developing a real-time ASL interpreter of not just static ASL images but also using computer vision to display text or voice while a person signs directly with someone on other end of the line.

Thanks to COVID-19 there has been an immediate acceleration of the adoption of video as a communication tool. No longer is it seen as a luxury form of communcation. With that in mind, I believe now is a great time to bridge the gap between these two communities and develop the tools necessary for the deaf community. 

## The Data




## Daily Journal Entry
**DAY 1:**
- wrote script to organize the train, validation, and test data into relative folders using os library. Became familiar with os functions for future use. 
- spending large portion of the my day reading and understanding keras API. Notes on things i've read:
    * image_dataset_from_directory is a tf.nightly function and I needed to pip install to use, this was to convert the complete path for each data set (train, valid, test) into grayscale, without doing it image by image. 
    * ImageDataGenerator is an augmentation, data generator that allows you to make subtle changes that can occure in real situtions without needing to go throuhg the arduous process of collecting new images.
        * rotation_range; not used in initial train
        * shear_range; not used in initial train
        * hori & vert flip; not used in initial train
        * fill_mode; not used in initial train 
        * and many more... above are some I will use when iterating through my model
- built the first baseline CNN model: (still grappling with what each layer in CNN does)
    * Sequential model
    * 3 Conv2D hidden layers
        * first 2 layers had filter=32, kernel_size=(5,5), activation='relu' and used maxpooling window pool_size=(2,2)
        * third layer used filter=64, kernel_size=(5,5), activation='relu' and used maxpooling window pool_size=(2,2)
    * applying reshaping of the data using flatten
    * applying 2 Dense layers 
        * first dense layer used unit=24, activation=relu.
        * second dense layer used unit=24, activation=tanh.
        * learned that using a unit=(less than # of classes) will return imcompatiable shape size [batch_size, unit] vs. [batch_size, classes], this comparison occurs between the train and validation set. 
    * applied a dropout rate between the 2 dense layers dropout=.5, helps prevent overfitting. 

- compiled the model using the following:
    * optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'
- finally I fit the model and used epoch=2 and the results were TERRIBLE!! (batch_size was 100)
- **1st attempt:**
    RESULTS:
    * Epoch 1/2 60/60 - 246s 4s/step - loss: nan - accuracy: 0.0530 - val_loss: nan - val_accuracy: 0.0536
    * Epoch 2/2 60/60 - 249s 4s/step - loss: nan - accuracy: 0.0539 - val_loss: nan - val_accuracy: 0.0536
- **2nd attempt:**
    * changed all Conv2D layers kernel_size=(2,2)
    * changed 2nd dense layer activation='softmax'
    RESULTS:
    * 60/60 - 118s 2s/step - loss: 11.0343 - accuracy: 0.0525 - val_loss: 3.1642 - val_accuracy: 0.0536
    60/60 - 118s 2s/step - loss: 3.1604 - accuracy: 0.0500 - val_loss: 3.1510 - val_accuracy: 0.0536
-**3rd attempt:**
    * changed epoch to 5, did not yield different results. Exactly the same val_accuracy with slight decrease in loss.   
- Training is taking over 1.5minute per epoch, best I look into using GPU with AWS EC2 virtual machine. 

#### Predict the letter of an image in American Sign Language (ASL)
**Prediction type:** Categorical<br>
**Data type:** Stationary/Images<br>
**Source:** Open source sites with ASL images already populated (GitHub), myself, and google images. Already have a large dataset from Microsoft from an ASL project they started in 2019. <br>
**Observations/features:** Data would include images of all 26 letters in alphabet from A-Z (some letters like j & z require movement). Ideally have 10 images of each letter with different backgrounds to train model.

**Summary:** Using the features provide within the data described above, using Neural Nets/Image Processing predict what letter of the alphabet is displayed in the image. Being raised by deaf parents I have an intimate relationship with the need for ASL to be more interpretable for the masses. This project will be broken down into distinct phases intended to tackle real-world issues for the deaf community. This phase will focus on developing a basic model that accurately predicts what letter of the alphabet a person is signing.

Project will be split into several phases:

**Phase I:** Collect static images of alphabet, create CNN/RFC models, train, evaluate, and test models on static images. - Capstone 2.

**Phase II:** create CNN model on non static images, train, evaluate, and test models.

**Phase III:** develop external site and/or app to share with close friends and family to have them collect new data based off displayed words and have a gamification process of the training.

**Phase IV:** TBD

data collect: word shown on screen, person signs the word, can do as many of these as they'd like.
data validate: option to validate others signs. 

create a positive feedback loop in which data fed in allows user to gain incentives. 


