

#CNN
from keras.models import Sequential
#sequential is used to initialize the neural network
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D  #to make the model spatial orientation proof
from keras.layers import Flatten, Dropout
#flatten forms input feature vector of feature maps(categories) formed after max pooling
from keras.layers import Dense #to add the feature vector in ANN
from keras.preprocessing import image
import cv2
import numpy as np
#initializing the cnn
classifier=Sequential()

# step 1- convolution. We will form a layer with all feature maps.
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#rectifier activation function is used to remove any negative value of pixels and to make the image non-linear

#step 2- pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# step 1- convolution. We will form a layer with all feature maps.
classifier.add(Convolution2D(32,3,3,activation='relu'))
#rectifier activation function is used to remove any negative value of pixels and to make the image non-linear

#step 2- pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Dropout Layers can be an easy and effective way to prevent overfitting in your models.
# A dropout layer randomly drops some of the connections between layers.
# This helps to prevent overfitting, because if a connection is dropped, the network is forced to Luckily, with keras it’s really easy to add a dropout layer.

#classifier.add(Dropout(0.8))
#step 3- Flattening
classifier.add(Flatten())

# The first layer of neural network has been formed.

#step 4- Full connection
#classifier.add(Dense(output_dim=128,activation='relu'))
#classifier.add(Dense(output_dim=128,activation='relu'))
#classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=26,activation='softmax'))
#Here we have 2 categories so we are using sigmod. Otherwise, we would have used softmax

#compiling the CNN using stochastic gradient descent
#rms=keras.optimizers.RMSprop(learning_rate=0.01,rho=0.9)

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#since there is binary classification, we choose binary crossentropy, otherwise, we will chose categorical cross entropy

#We will do image augmentation to training set. We will take batched of training data and will transform them. like 
#rotation, rescaling, zooming, shering, flips etc
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen.flow_from_directory()
training_set = train_datagen.flow_from_directory('data_a/training',
                                                target_size=(64, 64),
                                                batch_size=64,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('data_a/test',
                                            target_size=(64, 64),
                                            batch_size=128,  
                                            class_mode='categorical')
from keras.callbacks import History
history=History()

classifier.fit_generator(training_set,
                        samples_per_epoch=160069,
                        epochs=10,callbacks=[history],
                        validation_data=test_set,
                        nb_val_samples=3684)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

########################################################################
#   IP

cap=cv2.VideoCapture(0)
cv2.namedWindow('tracker')

def track(x):
    pass
mask=None

cv2.createTrackbar('LH','tracker',0,255,track)
cv2.createTrackbar('LS','tracker',0,255,track)
cv2.createTrackbar('LV','tracker',0,255,track)
cv2.createTrackbar('UH','tracker',255,255,track)
cv2.createTrackbar('US','tracker',255,255,track)
cv2.createTrackbar('UV','tracker',255,255,track)
img = cv2.imread('opencv0.png')
img*=0
cv2.imshow('img',img)
p=0
q=0
f=0
f2=0
f1=0

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #lc=np.array([cv2.getTrackbarPos('LH','tracker'),cv2.getTrackbarPos('LS','tracker'),cv2.getTrackbarPos('LV','tracker')])
    #uc=np.array([cv2.getTrackbarPos('UH','tracker'),cv2.getTrackbarPos('US','tracker'),cv2.getTrackbarPos('UV','tracker')])
    #mask=cv2.inRange(hsv,lc,uc)
    mask=cv2.inRange(hsv,np.array([0,176,185]),np.array([97,255,255]))
    kernel = np.ones((2, 2), np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=3)
    mask=cv2.bilateralFilter(mask,3,15,75)
    contours,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c)>50:
            cv2.putText(frame,'Visible',(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            f=1
            (x,y,w,h)=cv2.boundingRect(c)
    if f==1:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,255),2)
    #cv2.drawContours(frame,contours,-1,(0,255,255),2)
    k = cv2.waitKey(33)
    if k==ord('s'):
        f1=1
        f2=0
        p=0
    if k == ord('z'):
        f2 = 1
        p=0
    if f==1 and f1==1 and f2==0:
        cv2.circle(img,(x,y),1,(255,255,255),6)
        if p!=0:
            cv2.line(img,(x,y),(p,q),(255,255,255),30)
        p = x
        q = y

    if f1==1 or f2==0:
        frame |= img
    cv2.imshow('mask',mask)
    cv2.imshow('img',img)
    cv2.imshow('camera',frame)

    if k==27:
        break
    if k==ord('a'):
        img=255-img
        cv2.imshow('final image',img)
        img1=cv2.resize(img,(128,128))
        cv2.imwrite('test.png',img1)
        test_img = image.load_img('test.png',target_size=(64,64))
        test_img = image.img_to_array(test_img)
        test_img=np.expand_dims(test_img,axis=0)
        res=classifier.predict(test_img)
        training_set.class_indices
        prediction='error'
        c=chr(65+int(np.where(res[0] == np.amax(res[0]))[0]))
        print(c)

        img*=0
        f1=0

cap.release()
cv2.destroyAllWindows()