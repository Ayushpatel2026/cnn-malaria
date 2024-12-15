
from cnnClassifier import logger
import tensorflow as tf
from cnnClassifier.utils.tools import parse_tfrecord_fn, save_json
from cnnClassifier.entity.config_entity import (ModelEvaluationConfig)
from datetime import datetime

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def load_test_data(self):

        logger.info("Loading test data")
        test_data = tf.data.TFRecordDataset(self.config.test_data_path)
        test_data = test_data.map(parse_tfrecord_fn)

        # batch it by 1 to use for evaluation
        test_data = test_data.batch(1)

        logger.info("Test data loaded successfully.")
        return test_data

    def evaluate_model(self):
        test_data = self.load_test_data()
        model = tf.keras.models.load_model(self.config.trained_model_path)
        
        logger.info("Evaluation started...")
        results = model.evaluate(test_data, verbose=0)
        metric_names = model.metrics_names  # Get the names of the metrics

        # Combine the metric names with their corresponding values
        labeled_results = {name: value for name, value in zip(metric_names, results)}

        # Log the labeled results
        logger.info(f"Evaluation Results: {labeled_results}")
        
        self.save_evaluation_results(labeled_results["loss"], labeled_results["accuracy"])

        logger.info("Evaluation completed successfully and results saved.")
    
    def save_evaluation_results(self, test_loss, test_accuracy):
        evaluation_details = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_parameters": {
                "batch_size": self.config.all_params.BATCH_SIZE,
                "learning_rate": self.config.all_params.INITIAL_LEARNING_RATE,
                "epochs": self.config.all_params.EPOCHS,
                "dense_layer_one_size": self.config.all_params.DENSE_LAYER_ONE_SIZE,
                "dense_layer_two_size": self.config.all_params.DENSE_LAYER_TWO_SIZE,
                "filters": self.config.all_params.FILTERS,
                "kernel_size": self.config.all_params.KERNEL_SIZE,
                "stride_length": self.config.all_params.STRIDE_LENGTH,
                "pool_size": self.config.all_params.POOL_SIZE,
                "optimizer": "Adam",
                "loss_function": "BinaryCrossentropy",
                "activation": "relu",
                "data_augmentation": self.config.AUGMENT_DATA,
            },
            "evaluation_metrics": {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        }

        save_json(self.config.save_eval_path, evaluation_details)