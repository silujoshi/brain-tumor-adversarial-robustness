import torch
import os
from src.dataset import get_test_dataloader
from src.model import get_medical_cnn
from src.evaluate import evaluate_model

# from config import Config


def run_test(config):
    test_loader, classes = get_test_dataloader(config.DATA_PATH, config.BATCH_SIZE, config.IMAGE_SIZE)
    model = get_medical_cnn(len(classes), config.MODEL_NAME).to(config.DEVICE)
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Successfully loaded model weights from {config.MODEL_SAVE_PATH}")
    else:
        print(f"Warning: No saved model found at {config.MODEL_SAVE_PATH}.")

    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    print("\n--- Test Dataset Evaluation Results ---")
    return evaluate_model(all_targets, all_preds, classes)

if __name__ == '__main__':
    run_test()