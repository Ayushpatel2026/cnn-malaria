import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier import logger
import tensorflow as tf
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    def predict(self):
        # load model
        model = load_model('model/malaria_model.h5')

        logger.info("Predicting image class...")

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224), color_mode = "rgb")
        test_image = image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image, axis = 0)
        logger.info("Shape: {}".format(test_image.shape))
        result = model.predict(test_image)
        logger.info("Result: {}".format(result))

        if result[0] > 0.5:
            prediction = 'Uninfected'
            return [{ "image" : prediction}]
        else:
            prediction = 'Parasitized'
            return [{ "image" : prediction}]