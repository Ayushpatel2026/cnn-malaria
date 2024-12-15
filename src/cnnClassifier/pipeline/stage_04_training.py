from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.training import TrainingModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model Stage"


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        train_model = TrainingModel(training_config)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e