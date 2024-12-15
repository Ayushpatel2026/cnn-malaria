import os
from cnnClassifier import logger
import tensorflow as tf
from cnnClassifier.entity.config_entity import (TrainingConfig)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC

class TrainingModel:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trained_model = None
        self.metrics = [TruePositives(name='tp'), FalsePositives(name='fp'),
           TrueNegatives(name='tn'), FalseNegatives(name='fn'),
           BinaryAccuracy(name='accuracy'), Precision(name='precision'),
           Recall(name='recall'), AUC(name='auc')]

    def load_train_val_data(self):
        ## TODO
        pass
    
    def train_model(self):
        ## TODO
        pass

    def save_train_model(self):
        # TODO
        pass