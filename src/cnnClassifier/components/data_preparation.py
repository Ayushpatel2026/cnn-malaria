import os
from cnnClassifier import logger
import tensorflow_datasets as tfds
import tensorflow as tf
from cnnClassifier.entity.config_entity import (DataPreparationConfig)
from tensorflow.keras.layers import RandomFlip, RandomRotation
from cnnClassifier.utils.tools import parse_tfrecord_fn, save_as_tfrecord_batch, save_as_tfrecord


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def load_dataset_tfds(self):
        """
        Loads the TensorFlow dataset from the local TFRecord file.
        """
        try:
            logger.info("Starting dataset load from TensorFlow Datasets...")

            # Load the dataset
            dataset = tf.data.TFRecordDataset(self.config.data_source)

            # Parse the serialized dataset
            parsed_dataset = dataset.map(parse_tfrecord_fn)

            logger.info("Dataset loaded successfully.")

            return parsed_dataset

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e

    def split_dataset(self, dataset, train_ratio, val_test_ratio):

        logger.info("Dataset size: ", self.config.DATASET_SIZE)

        # take from 0 to train_ratio*dataset_size
        train_dataset = dataset.take(int(train_ratio*self.config.DATASET_SIZE))

        val_test_dataset = dataset.skip(int(train_ratio*self.config.DATASET_SIZE))

        # after removing train, take val_ratio*dataset_size
        val_dataset = val_test_dataset.take(int(val_test_ratio*self.config.DATASET_SIZE))

        test_dataset = val_test_dataset.skip(int(val_test_ratio*self.config.DATASET_SIZE))

        logger.info("Dataset split successfully.")

        return train_dataset, val_dataset, test_dataset

    def _augment_layer(self, image, labels):
        augment_layers = tf.keras.Sequential([
            RandomRotation(factor=(0.25, 0.26),),
            RandomFlip(mode="horizontal")
        ])
        return augment_layers(image, training=True), labels
    
    def augment_dataset(self, dataset):
        return dataset.map(self._augment_layer)
        
    def _resize(self, image, label):
        # note that the datatype will be changed to float32 here
        image = tf.image.resize(image, (self.config.IM_SIZE, self.config.IM_SIZE))
        return image, label
    
    def resize_dataset(self, dataset):
        return dataset.map(self._resize)
    
    def batch_dataset(self, data_set):
        return data_set.batch(self.config.BATCH_SIZE)
    
    def shuffle_dataset(self, data_set):
        return data_set.shuffle(self.config.SHUFFLE_BUFFER_SIZE)

    def prefetch_dataset(self, data_set):
        return data_set.prefetch(tf.data.AUTOTUNE)
    
    def store_split_datasets(self, train_dataset, val_dataset, test_dataset):
        # prepare directories
        os.makedirs(self.config.root_dir, exist_ok=True)
        try:
            logger.info("Starting to store the split datasets...")

            # Save the datasets to the artifacts directory in TFRecord format
            save_as_tfrecord_batch(dataset=train_dataset, save_path=self.config.train_data_path)
            save_as_tfrecord_batch(dataset=val_dataset, save_path=self.config.val_data_path)
            save_as_tfrecord(dataset=test_dataset, save_path=self.config.test_data_path, dtype=tf.float32)

            logger.info("Datasets saved successfully.")

        except Exception as e:
            logger.error(f"Error during data preparation: {e}")
            raise e
