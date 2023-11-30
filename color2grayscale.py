'''
Project by Melody Llinas & Melody Vu

The following script will take our in-color image dataset and convert them all to grayscale
automatically. The images were taken from a monkey species dataset from kaggle 
(via https://www.google.com/url?q=https://www.kaggle.com/datasets/slothkong/10-monkey-species/&sa=D&source=docs&ust=1701357394325106&usg=AOvVaw3E7ABWaJSivMcEKtBl3ldz)

'''
import os

def nameFirstImage():
    directory = "/Users/Melody/Desktop/Projects/ML/imageRecolorization/images"
    for root, dirs, files, in os.walk(directory):
        print("root:", root)
        print("sub-folder:", dirs)
        print("inside contents:", files)
        print('----------------------------------')

nameFirstImage()