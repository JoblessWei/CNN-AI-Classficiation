import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 
from keras import Sequential, layers, metrics, regularizers
import cv2
from PIL import Image
from pathlib import Path

# Make sure all images are in RGB
# currentdir = os.getcwd()
# AIdir = os.path.join(currentdir,'ImageData','AiArtData')
# Realdir = os.path.join(currentdir,'ImageData','RealArt')
# datadir = [AIdir, Realdir]
# for images in Path(datadir):
#     img = img.convert("RGB")
#     img.save()


# Preprocess the data into size 32 batches, 256x256
data = keras.preprocessing.image_dataset_from_directory('ImageData', batch_size=32, image_size=(256, 256))

# Scale images to RGB values between 0 and 1
scaled_data = data.map(lambda x,y: (x/255, y))# X is the img, y is the label

# To iterate through data:
iterator =  tf.data.NumpyIterator(scaled_data)
batch = iterator.next()

# Train test and validation sets
DATASET_SIZE = len(scaled_data)
train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)


train = scaled_data.take(train_size)
val = scaled_data.skip(train_size).take(val_size)
test = scaled_data.skip(train_size + val_size).take(test_size)
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
print(test.as_numpy_iterator())


# Making the model
l2_reg = regularizers.l2(l2=0.01)
model = Sequential([
    # Data augmentation layers
    # layers.RandomFlip(mode='horizontal_and_vertical',input_shape=(256, 256, 3)),
    # layers.RandomRotation(0.2),
    # layers.RandomZoom(0.2),

    # CNN MODEL
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)), # Reduces the pixels by half by getting the max of 2 adjacent pixels
    layers.Dropout(0.25), # Prevents Overfitting by randomly "dropping out" (i.e., setting to zero) a number of output features of the layer during training
    
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),


    layers.Flatten(),

    # Fully connected layers
    # layers.Dense(128, activation='relu', kernel_regularizer=l2_reg),
    # layers.BatchNormalization(),
    # layers.Dropout(0.5),

    layers.Dense(256, activation='relu', kernel_regularizer=l2_reg),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(64, activation='relu', kernel_regularizer=l2_reg), # CHANGE 64 TO 512 or 1024 NEXT TIME, ALSO CHANGE EARLIER LAYERS
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Dense(1, activation='sigmoid') # Returns one value based on sigmoid curve (between 0 and 1)
])

# Learning rate Scheduler
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
optimizer = keras.optimizers.RMSprop()
model.compile(optimizer=optimizer,
              loss=keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.summary()
model.save('aidetector.keras')

logdir = 'logs'
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir= logdir)
history = model.fit(train, epochs=100, validation_data=val, callbacks=[lr_scheduler, tensorboard_callback])


fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label= 'loss')
plt.plot(history.history['val_loss'], color='orange', label= 'val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label= 'accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label= 'val_acc')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()


p = keras.metrics.Precision()
r = keras.metrics.Recall()
acc = keras.metrics.BinaryAccuracy()

# Evaluate performance
for batch in test.as_numpy_iterator():
    x , y = batch
    ypred = model.predict(x)
    p.update_state(y, ypred)
    r.update_state(y, ypred)
    acc.update_state(y, ypred)

print(f'Precision: {p.result().numpy()}, Recall: {r.result().numpy()}, Accuracy: {acc.result().numpy()}')




