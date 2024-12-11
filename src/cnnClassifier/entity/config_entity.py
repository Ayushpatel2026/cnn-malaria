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