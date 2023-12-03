import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# parameters
IMG = 200
IMG_SIZE = [IMG, IMG]
numOfClasses = 10
batchSize = 32
EPOCHS = 30

# building the model
model = tf.keras.models.Sequential ([

    #convolution layers
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG,IMG,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG,IMG,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMG,IMG,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMG,IMG,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    # flattening
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5), #prevent overfitting, randomly 50% of input units

    # fully connected layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(numOfClasses, activation='softmax')
])

print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# get the data
trainingFolder = "C:/Users/huydu/introtoml/monkey_class_proj/monkeyClassification/images/training"
validationFolder = "C:/Users/huydu/introtoml/monkey_class_proj/monkeyClassification/images/validation"

trainingGenerator = ImageDataGenerator( rescale = 1. / 255, #augment/enrich images to get more data
                                         rotation_range = 20,
                                         width_shift_range = 0.2,
                                         height_shift_range = 0.2,
                                         shear_range = 0.2,
                                         zoom_range = 0.2,
                                         horizontal_flip = True)

training_set = trainingGenerator.flow_from_directory( trainingFolder,
                                                      shuffle = True,
                                                      target_size = IMG_SIZE,
                                                      batch_size = batchSize,
                                                      class_mode = 'categorical')

validationGenerator = ImageDataGenerator( rescale = 1. / 255)

validation_set = validationGenerator.flow_from_directory( validationFolder,
                                                      shuffle = False, ####################
                                                      target_size = IMG_SIZE,
                                                      batch_size = batchSize,
                                                      class_mode = 'categorical')

# get step number (rounding up)
trainingSteps = np.ceil(training_set.samples / batchSize) 
validationSteps = np.ceil(validation_set.samples / batchSize) 

# store best model found during training
bestModel_file = "C:/Users/huydu/introtoml/monkey_class_proj/monkeyClassification/bestMonkeyModel.h5"
bestModel = ModelCheckpoint(bestModel_file, monitor='val_accuracy', verbose=1, save_best_only=True)

# train the model
history = model.fit( training_set,
                    validation_data = validation_set,
                    epochs = EPOCHS,
                    steps_per_epoch = trainingSteps,
                    validation_steps = validationSteps,
                    verbose = 1,
                    callbacks = [bestModel])

# evaluate the model
valResults = model.evaluate(validation_set)
print(valResults)
print(model.metrics_names)

# chart the results
trainAcc = history.history['accuracy']
validationAcc = history.history['val_accuracy']
trainLoss = history.history['loss']
validationLoss = history.history['val_loss']

actualEpochs = range(len(trainAcc))
print("Actual Epochs: "+ str(actualEpochs))

plt.plot(actualEpochs, trainAcc, 'r', label="Training Accuracy")
plt.plot(actualEpochs, validationAcc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs Epochs')

plt.show()
