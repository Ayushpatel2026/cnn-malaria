from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.training import TrainingModel
from cnnClassifier import logger

STAGE_NAME = "Training Stage"


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        train_model = TrainingModel(training_config)
        train_model.train_model()

        # move the trained model from artifacts to the models directory
        logger.info("Saving trained model...")
        train_model.save_train_model()
        logger.info("Trained model saved successfully.")

        train_model.copy_model(training_config.trained_model_path, 'model/lenet_model.h5')


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e