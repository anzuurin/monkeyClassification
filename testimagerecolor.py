'''
# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img
from sklearn.metrics import mean_squared_error

# Define the CNN model
model = Sequential()

# Encoder
model.add(Conv2D(64, (3, 3), input_shape=(height, width, 1), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

# Decoder
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Adjusted paths for training data and recolored output
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'Users/huydu/introtoml/imagerecolor_proj/imageRecolorization/images/bw',  # Path to black and white images
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='input',
    subset='training',
)

val_generator = datagen.flow_from_directory(
    'Users/huydu/introtoml/imagerecolor_proj/imageRecolorization/images/bw',  # Path to black and white images
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='input',
    subset='validation',
)

# Train the model
model.fit(train_generator, epochs=num_epochs, validation_data=val_generator)

# Evaluate the model on the validation set
validation_generator = datagen.flow_from_directory(
    'Users/huydu/introtoml/imagerecolor_proj/imageRecolorization/images/color',  # Path to original colored images for validation
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='input',
    subset='validation',
    shuffle=False  # Important: Ensure no shuffling to maintain order for comparison
)

# Predict recolored images
recolored_predictions = model.predict(validation_generator)

# Save the recolored images in JPG format
output_dir = 'Users/huydu/introtoml/imagerecolor_proj/imageRecolorization/images/recolored'
os.makedirs(output_dir, exist_ok=True)

for i, recolored_image in enumerate(recolored_predictions):
    output_path = os.path.join(output_dir, f'recolored_{i}.jpg')
    save_img(output_path, recolored_image)
    print(f'Recolored image saved: {output_path}')

# Compute MSE
mse = mean_squared_error(validation_generator.next()[0], recolored_predictions)
print(f'Mean Squared Error: {mse:.4f}')
'''