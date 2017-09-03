import csv
import cv2
import numpy as np

#############################
#      Data Preparation     #
#############################

#Read in the adresses of the training data
samples = []
with open('C://DrivingData/Track_1_2_Loops/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    
#Shuffle data once
from sklearn.utils import shuffle
shuffle(samples)

#Define the train valid split
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Training samples: ' +str(6*len(train_samples)))
print('Validation samples: ' +str(6*len(validation_samples)))

#For each sample 6 training samples are generated
#Left camera image, right camera image and the center camera image (additionaly each image is flipped)
#The Steering angle is adapted according to the correction term 
#For the correction term we choose 0.02 (see 'correction_term.ipynb' for an experimental justification of this choice)
def generate_6(samples, batch_size = 10,correction_term = 0.02):
    num_samples = len(samples)
    shuffle(samples)
    while 1: 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
               
                img_left = cv2.imread(batch_sample[1])
                left_angle = center_angle+correction_term
                
                img_right = cv2.imread(batch_sample[2])
                right_angle = center_angle-correction_term   
        
                images.append(center_image)
                angles.append(center_angle)

                images.append(img_left)
                angles.append(left_angle)

                images.append(img_right)
                angles.append(right_angle)
                                
                #Add all flipped versions
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                
                images.append(np.fliplr(img_left))
                angles.append(-left_angle)

                images.append(np.fliplr(img_right))
                angles.append(-right_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train,y_train

#Initialize the data generators
training_batch_size = 10
validation_batch_size = 10

train_generator = generate_6(train_samples,training_batch_size)
validation_generator = generate_6(validation_samples,training_batch_size)

#############################
#      Model Definition     #
#############################

#Define the neural network model using Keras (we used Tensorflow as backend)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D, Conv2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0)-0.5))
model.add(AveragePooling2D(pool_size = 2))
model.add(Conv2D(5,(3,3),activation = 'relu'))
model.add(Conv2D(5,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(5,(3,3),activation = 'relu'))
model.add(Conv2D(5,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(50,activation = 'relu'))
model.add(Dense(25,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1))


s_p_e_t = len(train_samples)/training_batch_size
s_p_e_v = len(validation_samples)/validation_batch_size

#############################
#    Training the Model     #
#############################

model.compile(loss = 'mse', optimizer = 'adam')

#Save the model after each epoch
checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.6f}.h5',  monitor='val_loss')

model.fit_generator(train_generator, steps_per_epoch = s_p_e_t, epochs = 20,verbose = 1, callbacks = [checkpointer], validation_data=validation_generator, validation_steps=s_p_e_v)