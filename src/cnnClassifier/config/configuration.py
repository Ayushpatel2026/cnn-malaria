from cnnClassifier.constants import *
import os
from cnnClassifier.utils.tools import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import DataIngestionConfig, DataPreparationConfig
                                    


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
            IM_SIZE= config.IM_SIZE,
            BATCH_SIZE= config.BATCH_SIZE,
            SHUFFLE_BUFFER_SIZE= config.SHUFFLE_BUFFER_SIZE,
            train_data_path= config.train_data_path,
            val_data_path= config.val_data_path,
            test_data_path= config.test_data_path,
            TRAIN_SPLIT= config.TRAIN_SPLIT,
            DATASET_SIZE= config.DATASET_SIZE,
        )
        return date_preparation_config
    