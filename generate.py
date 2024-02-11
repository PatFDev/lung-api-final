import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configurable parameters
BASE_DIR = 'chest_xray 2'  # Dataset directory
TARGET_SIZE = (150, 150)  # Target size for images
BATCH_SIZE = 64  # Adjusted batch size
EPOCHS = 30  # Increased number of epochs
LEARNING_RATE = 0.0001  # Initial learning rate
REGULARIZATION = 0.001  # regularization

# Directories
train_dir = os.path.join(BASE_DIR, 'train')
validation_dir = os.path.join(BASE_DIR, 'val')
test_dir = os.path.join(BASE_DIR, 'test')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# Model creation with adjustments
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(REGULARIZATION)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(REGULARIZATION)),  
        Dropout(0.5),  
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

model = create_model((150, 150, 3))

callbacks = [
    ModelCheckpoint(filepath='best_model_improved', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),  
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001, verbose=1),  
]

# Training
try:
    model.fit(
    train_generator,
    steps_per_epoch=np.ceil(train_generator.samples / BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=np.ceil(validation_generator.samples / BATCH_SIZE),
    callbacks=callbacks,
    verbose=2
)
except Exception as e:
    print(f"Error during training: {e}")

# Evaluation
try:
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    loss, accuracy = model.evaluate(test_generator)
    model.save('final_model')
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")
except Exception as e:
    print(f"Error during evaluation: {e}")
