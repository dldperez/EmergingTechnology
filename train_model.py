import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 4

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\Dean\OneDrive\Desktop\BananaRipenessApp\pythonProject\BananaRipeness\train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    r"C:\Users\Dean\OneDrive\Desktop\BananaRipenessApp\pythonProject\BananaRipeness\valid",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Build CNN model using Rectified Linear Unit
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

import pickle

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit( train_generator, validation_data=valid_generator, epochs=20)

with open('training_history.pkl','wb') as f:
    pickle.dump(history.history, f)

# Saving trained model
model.save('banana_ripeness_model.h5')
print("Model saved as banana_ripeness_model.h5")


