from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    AUGMENT_DATA: bool
    IM_SIZE: int
    BATCH_SIZE: int
    SHUFFLE_BUFFER_SIZE: int
    data_source: Path
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    TRAIN_SPLIT: float
    DATASET_SIZE: int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path 
    KERNEL_SIZE: int
    STRIDE_LENGTH: int
    FILTERS: int
    POOL_SIZE: int
    DENSE_LAYER_ONE_SIZE: int
    DENSE_LAYER_TWO_SIZE: int
    OUTPUT_CLASSES: int
    INPUT_SIZE: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    EPOCHS: int
    INITIAL_LEARNING_RATE: float
    BATCH_SIZE: int
    train_data_path: Path
    val_data_path: Path
    base_model_path: Path
    LEARNING_RATE_PATIENCE: int
    EARLY_STOPPING_PATIENCE: int
    LEARNING_RATE_FACTOR: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    trained_model_path: Path
    test_data_path: Path
    save_eval_path: Path