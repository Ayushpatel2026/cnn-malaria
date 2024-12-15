import os
import shutil
from cnnClassifier import logger
import tensorflow as tf
from cnnClassifier.utils.tools import parse_tfrecord_fn
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
        # data is saved in tfrecord format
        logger.info("Loading train and validation data...")
        train_data = tf.data.TFRecordDataset(self.config.train_data_path)
        val_data = tf.data.TFRecordDataset(self.config.val_data_path)

        parsed_train_data = train_data.map(parse_tfrecord_fn)

        parsed_train_data = parsed_train_data.batch(self.config.BATCH_SIZE)

        parsed_val_data = val_data.map(parse_tfrecord_fn)
        parsed_val_data = parsed_val_data.batch(self.config.BATCH_SIZE)

        logger.info("Data loaded successfully.")

        return parsed_train_data, parsed_val_data
    
    def train_model(self):
        # load base model

        logger.info("Loading base model...")
        lenet_model = tf.keras.models.load_model(self.config.base_model_path)
        logger.info("Base model loaded successfully.")

        self.trained_model = lenet_model

        self.trained_model.compile(optimizer=Adam(learning_rate=self.config.INITIAL_LEARNING_RATE),
                                   loss=BinaryCrossentropy(),
                                   metrics=self.metrics)
        logger.info("Model compiled successfully.")

        train_data, val_data = self.load_train_val_data()
        callbacks = CustomCallBacks(self.config)
        epoch_logger = EpochLogger()

        logger.info("Training model...")
        self.trained_model.fit(train_data, epochs=self.config.EPOCHS, verbose=0, validation_data=val_data, 
                               callbacks=[callbacks.get_early_stopping(), callbacks.get_learning_rate_scheduler(), epoch_logger])

    def save_train_model(self):
        self.trained_model.save(self.config.trained_model_path)

    def copy_model(self, source_path, destination_path):
        """
        Copies the trained model from source_path to destination_path.
        """
        try:
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy the model
            shutil.copy(source_path, destination_path)
            logger.info(f"Model copied successfully from {source_path} to {destination_path}.")
        except Exception as e:
            logger.error(f"Failed to copy the model: {e}")

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
    
    def scheduler(self, epoch, learning_rate):
        if epoch <= self.config.LEARNING_RATE_PATIENCE:
            return learning_rate
        else:
            return float(learning_rate * tf.math.exp(-self.config.LEARNING_RATE_FACTOR))
        
class EpochLogger(Callback):
    def __init__(self):
        super(EpochLogger, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs metrics and other relevant information after each epoch.
        """
        logs = logs or {}
        logger.info(f"Epoch {epoch + 1}:")
        for key, value in logs.items():
            logger.info(f"    {key}: {value:.4f}")
