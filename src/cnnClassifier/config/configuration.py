from cnnClassifier.constants import *
import os
from cnnClassifier.utils.tools import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, DataPreparationConfig, PrepareBaseModelConfig, TrainingConfig, ModelEvaluationConfig
                                    


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config
    
    def get_data_preparation_config(self) -> DataPreparationConfig:
        # TODO
        config = self.config.data_preparation
        create_directories([config.root_dir])
        date_preparation_config = DataPreparationConfig(
            AUGMENT_DATA= config.AUGMENT_DATA,
            root_dir = config.root_dir,
            data_source = config.data_source,
            IM_SIZE= self.params.IMAGE_SIZE,
            BATCH_SIZE= self.params.BATCH_SIZE,
            SHUFFLE_BUFFER_SIZE= self.params.SHUFFLE_BUFFER_SIZE,
            train_data_path= config.train_data_path,
            val_data_path= config.val_data_path,
            test_data_path= config.test_data_path,
            TRAIN_SPLIT= config.TRAIN_SPLIT,
            DATASET_SIZE= self.params.DATASET_SIZE,
        )
        return date_preparation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = config.root_dir,
            base_model_path= config.base_model_path,
            KERNEL_SIZE= self.params.KERNEL_SIZE,
            STRIDE_LENGTH= self.params.STRIDE_LENGTH,
            FILTERS= self.params.FILTERS,
            POOL_SIZE= self.params.POOL_SIZE,
            DENSE_LAYER_ONE_SIZE= self.params.DENSE_LAYER_ONE_SIZE,
            DENSE_LAYER_TWO_SIZE= self.params.DENSE_LAYER_TWO_SIZE,
            OUTPUT_CLASSES= self.params.OUTPUT_CLASSES,
            INPUT_SIZE= self.params.IMAGE_SIZE
        )
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        training_config = TrainingConfig(
            root_dir = config.root_dir,
            trained_model_path = config.trained_model_path,
            EPOCHS = self.params.EPOCHS,
            INITIAL_LEARNING_RATE = self.params.INITIAL_LEARNING_RATE,
            BATCH_SIZE = self.params.BATCH_SIZE,
            train_data_path = self.config.data_preparation.train_data_path,
            val_data_path = self.config.data_preparation.val_data_path,
            base_model_path = self.config.prepare_base_model.base_model_path,
            EARLY_STOPPING_PATIENCE = self.params.EARLY_STOPPING_PATIENCE,
            LEARNING_RATE_PATIENCE = self.params.LEARNING_RATE_PATIENCE,
            LEARNING_RATE_FACTOR = self.params.LEARNING_RATE_FACTOR
        )
        return training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        model_evaluation_config = ModelEvaluationConfig(
            trained_model_path = config.trained_model_path,
            test_data_path = config.test_data_path,
            save_eval_path = config.save_eval_path,
            all_params = self.params,
            AUGMENT_DATA = self.config.data_preparation.AUGMENT_DATA,
        )
        return model_evaluation_config