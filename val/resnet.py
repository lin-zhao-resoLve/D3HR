import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import shutil
from PIL import Image
import random
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def load_mapping(mapping_file):
    wnid_to_index = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            wnid, index = line.strip().split('\t')
            wnid_to_index[wnid] = int(index)
    return wnid_to_index

def create_random_subset(source_path, destination_path, num_images=10):
    os.makedirs(destination_path, exist_ok=True)

    for class_name in os.listdir(source_path):
        class_source_path = os.path.join(source_path, class_name, 'images')
        class_destination_path = os.path.join(destination_path, class_name)
        
        # Create the class directory in the destination if it doesn't exist
        os.makedirs(class_destination_path, exist_ok=True)
        
        # Get all images in the class directory
        images = [img for img in os.listdir(class_source_path) if img.endswith(('.png', '.jpg', '.JPEG'))]
        
        # Randomly select specified number of images
        selected_images = random.sample(images, min(num_images, len(images)))
        
        # Copy selected images to the new location
        for img_name in selected_images:
            src_img_path = os.path.join(class_source_path, img_name)
            dest_img_path = os.path.join(class_destination_path, img_name)
            shutil.copy2(src_img_path, dest_img_path)
    
    print(f"Random selection of {num_images} images per class completed and saved in {destination_path}")


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mapping_file, split='train', transform=None, random_selection=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load class mapping
        self.wnid_to_index = load_mapping(mapping_file)
        
        if split == 'train':
            self._load_train_data(random_selection)
        elif split == 'val':
            self._load_val_data()
        else:  # test
            self._load_test_data()

    def _load_train_data(self, random_selection):
        class_dirs = sorted(os.listdir(self.root_dir))
        
        for class_name in class_dirs:
            if class_name not in self.wnid_to_index:
                continue
                
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            image_files = sorted(os.listdir(class_path))
            
            if random_selection:
                # Randomly select 10 images for this class
                image_files = random.sample(image_files, min(10, len(image_files)))
            
            for img_file in image_files:
                self.images.append(os.path.join(class_path, img_file))
                self.labels.append(self.wnid_to_index[class_name])

    def _load_val_data(self):
        # Load validation annotations
        val_annotations_path = os.path.join(os.path.dirname(self.root_dir),'val', 'val_annotations.txt')
        img_to_class = {}
        
        with open(val_annotations_path, 'r') as f:
            for line in f:
                img_file, class_name, *_ = line.strip().split('\t')
                img_to_class[img_file] = class_name
        
        images_dir = os.path.join(self.root_dir)
        image_files = sorted(os.listdir(images_dir))
        
        for img_file in image_files:
            if img_file in img_to_class and img_to_class[img_file] in self.wnid_to_index:
                self.images.append(os.path.join(images_dir, img_file))
                self.labels.append(self.wnid_to_index[img_to_class[img_file]])

    def _load_test_data(self):
        # Load test annotations
        test_annotations_path = os.path.join(os.path.dirname(self.root_dir), 'test', 'test_annotations.txt')
        img_to_class = {}
        
        with open(test_annotations_path, 'r') as f:
            for line in f:
                img_file, class_name, *_ = line.strip().split('\t')
                img_to_class[img_file] = class_name
        
        images_dir = os.path.join(self.root_dir, 'images')
        image_files = sorted(os.listdir(images_dir))
        
        for img_file in image_files:
            if img_file in img_to_class and img_to_class[img_file] in self.wnid_to_index:
                self.images.append(os.path.join(images_dir, img_file))
                self.labels.append(self.wnid_to_index[img_to_class[img_file]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (64, 64))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(is_training):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
def train_model(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    best_train_acc = 0.0  # 保存训练集上的最佳准确率
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 计算当前 epoch 的训练准确率
        train_acc = train_correct / train_total
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {100.*train_acc:.2f}%')
        
        # 保存模型检查点，如果训练准确率提升
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_train_acc': best_train_acc,
            }, checkpoint_path)
            print(f'Checkpoint saved with training accuracy: {100.*best_train_acc:.2f}%')

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
#     best_val_acc = 0.0
    
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         train_correct = 0
#         train_total = 0
        
#         for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             train_total += labels.size(0)
#             train_correct += predicted.eq(labels).sum().item()
        
#         # Validation phase
#         model.eval()
#         val_correct = 0
#         val_total = 0
#         val_top5_correct = 0
        
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
                
#                 # Top-1 accuracy
#                 _, predicted = outputs.max(1)
#                 val_total += labels.size(0)
#                 val_correct += predicted.eq(labels).sum().item()
                
#                 # Top-5 accuracy
#                 _, top5_pred = outputs.topk(5, 1, True, True)
#                 top5_pred = top5_pred.t()
#                 correct = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
#                 val_top5_correct += correct.any(dim=0).sum().item()
        
#         val_acc = val_correct / val_total
#         val_top5_acc = val_top5_correct / val_total
        
#         print(f'Epoch {epoch+1}:')
#         print(f'Training Loss: {running_loss/len(train_loader):.4f}')
#         print(f'Training Accuracy: {100.*train_correct/train_total:.2f}%')
#         print(f'Validation Top-1 Accuracy: {100.*val_acc:.2f}%')
#         print(f'Validation Top-5 Accuracy: {100.*val_top5_acc:.2f}%')
        
#         # Save checkpoint if validation accuracy improves
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_val_acc': best_val_acc,
#             }, checkpoint_path)
#             print(f'Checkpoint saved with validation accuracy: {100.*best_val_acc:.2f}%')

def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            top1_correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
            top5_correct += correct.any(dim=0).sum().item()
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    per_class_acc = 100. * class_correct / class_total
    
    return top1_acc, top5_acc, per_class_acc.cpu().numpy()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    mapping_file = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/tiny-imagenet-mapping.txt" 
    distilled_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/key50_far_images_inversion_train_vit_image_feature_CLS"
    val_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/val"
    # test_dir = "../../dataset/tiny-imagenet-200/test"
    train_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/train"
    # random_10_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/random_10_per_class"
    random_50_dir = "/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/random_50_per_class"
    # create_random_subset(train_dir, random_10_dir, num_images=10)
    create_random_subset(train_dir, random_50_dir, num_images=50)
    
    # Create datasets and dataloaders
    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)
    
    # Distilled dataset
    distilled_train_dataset = TinyImageNetDataset(distilled_dir, mapping_file, split='train', 
                                                transform=train_transform, 
                                                random_selection=False)
    
    # Random selection dataset
    random_train_dataset = TinyImageNetDataset(random_50_dir, mapping_file, split='train', 
                                             transform=train_transform, 
                                             random_selection=False)
    
    # Validation and test datasets
    val_dataset = TinyImageNetDataset(val_dir, mapping_file, split='val', 
                                    transform=eval_transform)
    # test_dataset = TinyImageNetDataset(test_dir, mapping_file, split='test', 
                                    #  transform=eval_transform)
    
    train_loaders = {
        'Distilled': DataLoader(distilled_train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=4),
        'Random': DataLoader(random_train_dataset, batch_size=batch_size, 
                           shuffle=True, num_workers=4)
    }
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, 
    #                        shuffle=False, num_workers=4)
    
    # Get number of classes from mapping
    num_classes = len(load_mapping(mapping_file))
    
    results = {}
    
    for dataset_type, train_loader in train_loaders.items():
        print(f"\nTraining with {dataset_type} dataset:")
        print(f"Number of training images: {len(train_loader.dataset)}")
        
        # Initialize model with correct number of classes
        model = models.resnet50(num_classes=num_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        checkpoint_path = f'checkpoint_{dataset_type.lower()}.pth'
        
        # Train model
        # train_model(model, train_loader, val_loader, criterion, optimizer, 
        #            num_epochs, device, checkpoint_path)
        train_model(model, train_loader, criterion, optimizer, 
                   num_epochs, device, checkpoint_path)
        
        # Load best checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        top1_acc, top5_acc, per_class_acc = evaluate_model(model, val_loader, 
                                                         device, num_classes=num_classes)
        
        results[dataset_type] = {
            'Top-1 Accuracy': top1_acc,
            'Top-5 Accuracy': top5_acc,
            'Per-class Accuracy (Mean)': np.mean(per_class_acc),
            'Per-class Accuracy (Std)': np.std(per_class_acc),
            'Per-class Accuracy (Min)': np.min(per_class_acc),
            'Per-class Accuracy (Max)': np.max(per_class_acc)
        }
    
    # Create results table
    results_df = pd.DataFrame(results).round(2)
    print("\nResults Comparison:")
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv('comparison_results.csv')

if __name__ == "__main__":
    main()