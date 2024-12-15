import os
from cnnClassifier import logger
import tensorflow as tf
from cnnClassifier.entity.config_entity import (TrainingConfig)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
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

class CustomCallBacks:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=self.config.LEARNING_RATE_PATIENCE, verbose=1,
            mode='auto', baseline=None, restore_best_weights=True
        )
        self.learning_rate_scheduler = LearningRateScheduler(self.scheduler, verbose=1)
    
    def get_early_stopping(self):
        return self.early_stopping
    
    def get_learning_rate_scheduler(self):
        return self.learning_rate_scheduler
    
    def scheduler(self, epoch, lr):
        if epoch <= self.config.LEARNING_RATE_PATIENCE:
            return lr
        else:
            return float(lr * tf.math.exp(-self.config.LEARNING_RATE_FACTOR))
