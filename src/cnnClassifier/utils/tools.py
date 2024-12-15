import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import tensorflow as tf

'''
This file contains all of the common utility functions that are used in 
the entire project.
'''


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path, evaluation_details):
    """save json data

    Args:
        path : path to json file
        evaluation_details : data to be saved in json file
    """
    # Create the file if it doesn't exist
    if not os.path.exists(path):
        logger.info("Creating a new JSON file.")
        with open(path, 'w') as json_file:
            json.dump([], json_file)  # Initialize with an empty list

    # Append the new evaluation details to the JSON file
    with open(path, 'r+') as json_file:
        try:
            # Load existing data
            data = json.load(json_file)
        except json.JSONDecodeError:
            data = []

        # Append new evaluation details
        data.append(evaluation_details)

        # Write back to the file
        json_file.seek(0)  # Move to the beginning of the file
        json.dump(data, json_file, indent=4)
    
    logger.info(f"Evaluation details saved successfully to {path}.")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def save_as_tfrecord(dataset, save_path, dtype=tf.uint8):
        """
        Saves the given dataset to a TFRecord file.
        """
        with tf.io.TFRecordWriter(save_path) as writer:
            for image, label in dataset:
                # Serialize image and label into a TFRecord format
                if dtype == tf.float32:
                    image = tf.cast(image * 255.0, tf.uint8)

                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def save_as_tfrecord_batch(dataset, save_path, dtype=tf.float32):
        """
        Saves the given dataset to a TFRecord file.
        """
        with tf.io.TFRecordWriter(save_path) as writer:
            for batch in dataset:
                images, labels = batch
                for image, label in zip(images, labels):
                    # Serialize image and label into a TFRecord format
                    if dtype == tf.float32:
                        image = tf.cast(image * 255.0, tf.uint8)

                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

def parse_tfrecord_fn(serialized_data):
    """
    Parses the serialized data into image and label.
    """
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(serialized_data, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    label = example['label']
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())