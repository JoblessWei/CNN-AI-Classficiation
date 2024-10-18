import cv2
import tensorflow as tf
import keras 
import numpy as np


# 1 = Real 
# 0 = AI
img = cv2.imread('realpanda.jpg')
resize = tf.image.resize(img, (256,256)) 

new_model = keras.saving.load_model('aidetector.keras')
ypredNew = new_model.predict(np.expand_dims(resize/255, 0)) # exand_dims puts the numpy array into another list, making sure it fits the shape
print(f"Prediction: {ypredNew}")
if ypredNew > 0.5:
    print("REAL IMAGE")
else:
    print("AI IMAGE")