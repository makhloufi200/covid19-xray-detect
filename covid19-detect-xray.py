from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

data = []
labels = []

#Code for making images into array -
covids=os.listdir("Covid")
for covid in covids:
    #print(cat)
    imag = cv2.imread("Covid/"+covid)
    #print(imag)
    img_from_ar = Image.fromarray(imag)
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

normals=os.listdir("Normal")
for normal in normals:
    imag=cv2.imread("Normal/"+normal)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)



#Since the “data” and “labels” are normal array , convert them to numpy arrays-
images=np.array(data)
labels=np.array(labels)

#Now save these numpy arrays so that you dont need to do this image manipulation again.
np.save("images",images)
np.save("labels",labels)

#Load the arrays ( Optional : Required only if you have
# closed your jupyter notebook after saving numpy array )
images=np.load("images.npy")
labels=np.load("labels.npy")

#Now shuffle the “animals” and “labels” set so that you get
# good mixture when you separate the dataset into train and test
s=np.arange(images.shape[0])
np.random.shuffle(s)
images=images[s]
labels=labels[s]

#Make a variable num_classes which is the total number of images
# categories and a variable data_length which is size of dataset
num_classes=len(np.unique(labels))
data_length=len(images)

#Divide data into test and train
#Take 90% of data in train set and 10% in test set
(x_train,x_test)=images[(int)(0.1*data_length):],images[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

#Divide labels into test and train
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

#Make labels into One Hot Encoding
#One hot encoding
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

#make model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

#Step 4 — Train the model
#Congrats on reaching upto this step. Your all work is over. Now just train the model and wait patiently.
model.fit(x_train,y_train,batch_size=50
          ,epochs=100,verbose=1)

#Step 5 — Test the model
#Use model.evaluate to see how model work on test set
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

#Step 6 — Predicting on single images
#If you want to predict on a single image use this code-

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def get_image_result(label):
    if label==0:
        return "Covid"
    if label==1:
        return "Normal"

def predict_image(file):
		print("Predicting .................................")
		ar=convert_to_array(file)
		ar=ar/255
		label=1
		a=[]
		a.append(ar)
		a=np.array(a)
		score=model.predict(a,verbose=1)
		print(score)
		label_index=np.argmax(score)
		print(label_index)
		acc=np.max(score)
		xray_image=get_image_result(label_index)
		print(xray_image)
		print("The predicted X-Ray Image is a "+xray_image+" with accuracy =    "+str(acc))
		# release resources
		img = cv2.imread(file)
		position = ((int) (img.shape[1]/2 - 268/2), (int) (img.shape[0]/2 - 36/2))
		position1 = ((int) (img.shape[1]/2 - 268/2), (int) ((img.shape[0]/2 - 36/2)+80))
		cv2.putText(img,"With: " +'{:2.2%}'.format(acc),position1,
             cv2.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255),2)
		cv2.putText(img, xray_image,position,
             cv2.FONT_HERSHEY_SIMPLEX, 3, (209, 80, 0, 255),2)
		cv2.imshow("frame",img)
		cv2.waitKey()
		cv2.destroyAllWindows()

#predict_image("n.jpeg")
#predict_image("v.png")
predict_image("p.jpeg")
