'''
The following model is what we will train and use to classify images
of monkeys.
'''
'''
The following model is what we will train and use to classify images
of monkeys.
'''
import os
import matplotlib.pyplot as plt
import random
import pandas as pd

trainingImages = "/Users/Melody/Desktop/Projects/ML/monkeyClassification/images/training"
validationImages = "/Users/Melody/Desktop/Projects/ML/monkeyClassification/images/validation"

# accessing the monkey_labels.txt file
columns = ["Label","Latin Name","Common Name","Train","Valid"]

df = pd.read_csv("/Users/Melody/Desktop/Projects/ML/monkeyClassification/images/monkey_labels.txt",names=columns,skiprows=1)

# formatting labels
df['Label'] = df['Label'].str.strip()
df['Latin Name'] = df['Latin Name'].replace("\t","")
df['Latin Name'] = df['Latin Name'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()

df = df.set_index("Label")

# This prints the monkey label file in an organized way
# print(df)

monkeyDict = df["Common Name"] # use this for monkey species name
# print(monkeyDict)

# displaying 6 random images of monkeys per row with their corresponding species label above it
def displayImages(imgFolder):
    folderList = [f for f in os.listdir(imgFolder) if not f.startswith('.')] # skips over .DS_Store file
    folderList.sort()

    numOfSpecies = len(folderList)
    numOfColumns = 6

    fig, ax = plt.subplots(numOfSpecies, numOfColumns, figsize=(2.5*numOfColumns, 0.75*numOfSpecies))

    for rowNum, folderClassItem in enumerate(folderList):
        path = os.path.join(imgFolder, folderClassItem)
        subDir = os.listdir(path)

        for i in range(numOfColumns):
            randomImg = random.choice(subDir)
            imgPath = os.path.join(path, randomImg)
            img = plt.imread(imgPath)
            speciesLabel = monkeyDict[folderClassItem]
            speciesLabel = speciesLabel[:15]
            
            ax[rowNum, i].set_title(speciesLabel, fontsize=8)
            ax[rowNum, i].imshow(img)
            ax[rowNum, i].axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    plt.show()

displayImages(trainingImages)