import os
import shutil
import random

def split_dataset(source_root, output_root="./data", train_split=0.7, val_split=0.15):
    # Set random seed for reproducibility
    random.seed(42)
    
    # Classes based on your folders
    classes = ['Brain Tumor', 'Healthy']
    
    splits = ['train', 'val', 'test']
    
    # Create target directories
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)
            
    for cls in classes:
        cls_path = os.path.join(source_root, cls)
        if not os.path.exists(cls_path):
            print(f"Warning: Class folder {cls_path} not found. Skipping.")
            continue
            
        images = os.listdir(cls_path)
        # Filter for image extensions
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        total_images = len(images)
        train_end = int(total_images * train_split)
        val_end = train_end + int(total_images * val_split)
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        dataset_splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        
        for split, img_list in dataset_splits.items():
            for img in img_list:
                src_file = os.path.join(cls_path, img)
                dst_file = os.path.join(output_root, split, cls, img)
                shutil.copyfile(src_file, dst_file)
                
        print(f"Class '{cls}': Copied {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test images.")

    print("\nDataset split successfully into './data/'!")

if __name__ == "__main__":
    raw_source = "./Brain Tumor Data Set/Brain Tumor Data Set"
    split_dataset(raw_source)