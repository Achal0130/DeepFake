import json
import os
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0

print('TensorFlow version: ', tf.__version__)

dataset_path = './split_dataset/'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

tmp_debug_path = './tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    batch_size=batch_size_num,
    class_mode="binary",
    shuffle=True
)

val_datagen = ImageDataGenerator(
    rescale=1/255
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    batch_size=batch_size_num,
    class_mode="binary",
    shuffle=True
)

test_datagen = ImageDataGenerator(
    rescale=1/255
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    batch_size=1,
    shuffle=False
)

efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'best_model.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

num_epochs = 20
history = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)
print(history.history)

# Load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# Generate predictions
test_generator.reset()
preds = best_model.predict(
    test_generator,
    verbose=1
)

# Create DataFrame to store predictions
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
print(test_results)
