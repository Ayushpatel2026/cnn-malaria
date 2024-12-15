
from cnnClassifier import logger
import tensorflow as tf
from cnnClassifier.utils.tools import parse_tfrecord_fn
from cnnClassifier.entity.config_entity import (ModelEvaluationConfig)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        

    