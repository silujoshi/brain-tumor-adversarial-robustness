import torch.nn as nn
from torchvision import models

def get_medical_cnn(num_classes, model_name="resnet50"):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Freeze early layers, fine-tune later blocks
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
    return model