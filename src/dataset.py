import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size, image_size):
    train_tf, val_tf = get_transforms(image_size)
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.classes

def get_test_dataloader(data_dir, batch_size, image_size):
    _, val_tf = get_transforms(image_size) # Reuse validation/test transforms (no data augmentation)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader, test_dataset.classes