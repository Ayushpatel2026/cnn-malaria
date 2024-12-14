import os
from cnnClassifier import logger
import tensorflow as tf
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig)
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense, Flatten, InputLayer

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        
    def build_model(self):
        logger.info("Building a model with following parameters: Kernel Size - {}\n Stride length - {}\n Filters - {}\n Pool Size - {} ".format(
                                       self.config.KERNEL_SIZE, 
                                       self.config.STRIDE_LENGTH, 
                                       self.config.FILTERS, 
                                       self.config.POOL_SIZE))
        
        logger.info(" Dense Layer One Size - {}\n Dense Layer Two Size - {}\n Output Classes - {}".format(
                self.config.DENSE_LAYER_ONE_SIZE, 
                self.config.DENSE_LAYER_TWO_SIZE, 
                self.config.OUTPUT_CLASSES))

        lenet_model = tf.keras.Sequential([
            InputLayer(input_shape=(self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3)),

            Conv2D(filters=self.config.FILTERS, 
                   kernel_size=self.config.KERNEL_SIZE, 
                   strides=self.config.STRIDE_LENGTH,
                   padding='valid', activation='relu',
                ),
            BatchNormalization(),
            MaxPool2D(pool_size=self.config.POOL_SIZE, strides=2*self.config.STRIDE_LENGTH),


            Conv2D(filters=self.config.FILTERS, 
                   kernel_size=self.config.KERNEL_SIZE, 
                   strides=self.config.STRIDE_LENGTH,
                   padding='valid', activation='relu'
                ),
            BatchNormalization(),
            MaxPool2D(pool_size=self.config.POOL_SIZE, strides=2*self.config.STRIDE_LENGTH),

            Flatten(),

            Dense(self.config.DENSE_LAYER_ONE_SIZE, activation="relu", ),
            BatchNormalization(),
            Dense(self.config.DENSE_LAYER_TWO_SIZE, activation="relu", ),
            BatchNormalization(),

            Dense(self.config.OUTPUT_CLASSES, activation="sigmoid"),
        ])

        logger.info("Model built successfully.")
        logger.info("Model Summary:".format(lenet_model.summary()))
        self.model = lenet_model
    
    def save_model(self):
        logger.info(f"Saving model to {self.config.base_model_path}")
        self.model.save(self.config.base_model_path)