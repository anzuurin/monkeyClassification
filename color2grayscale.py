'''
Project by Melody Llinas & Melody Vu

The following script will take our in-color image dataset and convert them all to grayscale
automatically. The images were taken from a monkey species dataset from kaggle 
(via https://www.google.com/url?q=https://www.kaggle.com/datasets/slothkong/10-monkey-species/&sa=D&source=docs&ust=1701357394325106&usg=AOvVaw3E7ABWaJSivMcEKtBl3ldz)

'''
import os # used for traversing the directories
from PIL import Image # used for image processing

def color2grayscale():
    color_directory = "/Users/Melody/Desktop/Projects/ML/imageRecolorization/images/color"
    # the walk function traverses the directory in a dfs way
    for root, dirs, files, in os.walk(color_directory):

        # Check if the current directory is one of the n0-n9 folders
        if root.endswith(tuple([f"n{i}" for i in range(10)])):
            # traverses through every image
            for file in files:
                if file.lower().endswith(('.jpg')):
                    
                    file_path = os.path.join(root, file) # path of the color image
                    bw_path = root.replace("/color/", "/bw/") # path of bw mirror directory
                    grayscale_path = os.path.join(bw_path, file) # path the grayscale image will go to inside bw mirror directory

                    # open the image
                    with Image.open(file_path) as img:
                        # convert to grayscale
                        grayscale_img = img.convert('L')
                        grayscale_img.save(grayscale_path)


color2grayscale()