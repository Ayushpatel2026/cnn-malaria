import os
from cnnClassifier import logger
import tensorflow_datasets as tfds
import tensorflow as tf
from cnnClassifier.entity.config_entity import (DataIngestionConfig)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_save_tfds(self):
        """
        Downloads the TensorFlow dataset and saves it in the artifacts directory.
        """
        try:
            logger.info("Starting dataset download from TensorFlow Datasets...")

            # Download the dataset
            dataset, dataset_info = tfds.load(
                'malaria', 
                with_info=True,
                as_supervised=True,
                shuffle_files=True,
                split=['train']
            )
            logger.info("Dataset downloaded successfully.")

            # Prepare directories
            os.makedirs(self.config.root_dir, exist_ok=True)
            save_path = self.config.local_data_file
            
            # Save the dataset to the artifacts directory in TFRecord format
            logger.info(f"Saving dataset to {save_path} as TFRecord...")
            self._save_as_tfrecord(dataset=dataset[0], save_path=save_path)

            # Log dataset info
            logger.info(f"Dataset Info: {dataset_info}")
            logger.info(f"Dataset saved successfully at {save_path}")

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise e

    @staticmethod
    def _save_as_tfrecord(self, dataset, save_path):
        """
        Saves the given dataset to a TFRecord file.
        """
        with tf.io.TFRecordWriter(save_path) as writer:
            for image, label in dataset:
                # Serialize image and label into a TFRecord format
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())