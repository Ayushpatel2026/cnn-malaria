from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preparation import DataPreparation
from cnnClassifier import logger

STAGE_NAME = "Data Preparation Stage"


class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(data_preparation_config)
        dataset = data_preparation.load_dataset_tfds()
        train_dataset, val_dataset, test_dataset = data_preparation.split_dataset(dataset, 
                                                                                  data_preparation_config.TRAIN_SPLIT, 
                                                                                  (1 - data_preparation_config.TRAIN_SPLIT)/2)
        train_dataset = data_preparation.resize_dataset(train_dataset)
        train_dataset = data_preparation.shuffle_dataset(train_dataset)
        
        if data_preparation_config.AUGMENT_DATA:
            train_dataset = data_preparation.augment_dataset(train_dataset)

        train_dataset = data_preparation.batch_dataset(train_dataset)
        train_dataset = data_preparation.prefetch_dataset(train_dataset)


        val_dataset = data_preparation.resize_dataset(val_dataset)
        val_dataset = data_preparation.shuffle_dataset(val_dataset)
        val_dataset = data_preparation.batch_dataset(val_dataset)
        val_dataset = data_preparation.prefetch_dataset(val_dataset)

        test_dataset = data_preparation.resize_dataset(test_dataset)

        data_preparation.store_split_datasets(train_dataset, val_dataset, test_dataset)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e