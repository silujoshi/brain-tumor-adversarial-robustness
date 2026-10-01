import os
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Set your dataset name here: "chest_xray", "ct_scan", or "brain_tumor"
    DATASET_NAME: str = "brain_tumor" 
    DATA_PATH: str ="./brain_tumor"
    BATCH_SIZE: int = 32
    EPOCHS: int = 25
    LEARNING_RATE: float = 1e-4
    IMAGE_SIZE: tuple = (224, 224)
    MODEL_NAME: str = "resnet50"
    WEIGHT_DECAY: float = 1e-3
    OUTPUT_DIR: str = "./output/braintumor"
    
    # Automatically populated fields
    NUM_CLASSES: int = field(init=False)
    MODEL_SAVE_PATH: str = field(init=False)
    DEVICE: torch.device = field(init=False)

    def __post_init__(self):
        # Define class mappings for each medical dataset
        dataset_class_map = {
            "chest_xray": 2,      
            "ct_scan": 2,         
            "brain_tumor": 2      
        }
        
        # Normalize string input
        key = self.DATASET_NAME.lower().strip()
        
        if key in dataset_class_map:
            self.NUM_CLASSES = dataset_class_map[key]
        else:
            raise ValueError(
                f"Invalid dataset name '{self.DATASET_NAME}'. "
                f"Choose from: {list(dataset_class_map.keys())}"
            )
        
        # Set up dynamic paths and hardware DEVICE
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.MODEL_SAVE_PATH = os.path.join(self.OUTPUT_DIR, f"best_{key}_model.pth")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")