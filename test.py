# import cv2
# vidcap = cv2.VideoCapture('autorun.mp4')
# success, image = vidcap.read()
# count = 1
# while success:
#   cv2.imwrite("run4/image_%d.jpg" % count, image)    
#   success, image = vidcap.read()
#   print('Saved image ', count)
#   count += 1
import numpy as np
import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('./Models/model_ref.h5')

model.save('Models/model_update.h5')