import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import random

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# parameters
imageSize = [200, 200]
batchSize = 32 

# get the best model from training
bestModel_file = "C:/Users/huydu/introtoml/monkey_class_proj/monkeyClassification/bestMonkeyModel.h5" #path to model file
model = load_model(bestModel_file)

# get validation data
validationFolder = "" #path to validation folder
validation_generator = ImageDataGenerator(rescale = 1. /255)
validation_set = validation_generator.flow_from_directory(validationFolder, shuffle = False, target_size = imageSize, 
                                                          batch_size = batchSize, class_mode = 'categorical')

# predict the species of the validation set using the model
predictions = model.predict(validation_set)
predictionResults = np.argmax(predictions, axis=1)
print(predictionResults)

# print out monkey_labels.txt
columns = ["Label", "Latin Name", "Common Name", "Train Images", "Validation Images"]
# df is a Data Frame containing the info from monkey_labels.txt
df = pd.read_csv("C:/Users/huydu/introtoml/monkey_class_proj/monkeyClassification/images/monkey_labels.txt", names=columns, skiprows=1) 
# trim stuff for formatting
df['Label'] = df['Label'].str.strip()
df['Latin Name'] = df['Latin Name'].replace("\t", "")
df['Latin Name'] = df['Latin Name'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()
df = df.set_index("Label")
print(df)

# print out random 30 pics with the predicted species next to real species
monkeyNames = df["Common Name"]
def compareResults():
    image_files = glob.glob(validationFolder + '/*/*.jpg')
    numRows = 5
    numCols = 6
    numPics = numRows * numCols

    fig, ax = plt.subplots(numRows, numCols, figsize=(3*numCols, 3*numRows))
    correct = 0

    for i in range(numPics):
        # pick random picture
        x = random.choice(image_files)
        xInd = image_files.index(x)
        xImage = plt.imread(x)

        # get prediction for this pic
        xPred = monkeyNames[predictionResults[xInd]]
        xPred = xPred[:7]

        # get real species for this pic
        xReal = monkeyNames[validation_set.classes[xInd]]
        xReal = xReal[:7]

        # if correct, record
        if(xPred == xReal):
            correct += 1
        
        xTitle = 'predicted: {} \nreal: {}'.format(xPred, xReal)
        plt.imshow(xImage)
        plt.title(xTitle)

    print(" -------------------------------------------------------------------------")
    print("Total Pictures: {} Predictions Correct: {}".format(numPics, correct))

    plt.show()

# run the function
compareResults()