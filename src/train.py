import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# from config import Config
from src.dataset import get_dataloaders
from src.model import get_medical_cnn
from src.evaluate import evaluate_model


def train(config):
    # 1. Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    train_loader, val_loader, classes = get_dataloaders(config.DATA_PATH, config.BATCH_SIZE, config.IMAGE_SIZE)
    model = get_medical_cnn(len(classes), config.MODEL_NAME).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        v_loss = val_loss / val_total
        v_acc = val_correct / val_total
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        scheduler.step(v_loss)
        
        # Save best model to Config path
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"-> Saved new best model at epoch {epoch+1} to {config.MODEL_SAVE_PATH}")
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Train Loss: {epoch_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")

    # Plotting code remains the same...
    return evaluate_model(all_targets, all_preds, classes)

if __name__ == '__main__':
    train()