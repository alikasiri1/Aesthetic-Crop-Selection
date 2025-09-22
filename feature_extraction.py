import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, resnet101
import json
import numpy as np
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class PreferenceDataset(Dataset):
    """Dataset class for loading preference pairs from Part 1 data"""
    
    def __init__(self, preference_file, image_folder=None, transform=None, color_mode='RGB'):
        """
        Args:
            preference_file: JSON file from Part 1
            image_folder: Folder containing original images (if None, assumes crops are stored)
            transform: Image transformations
            color_mode: 'RGB' or 'L' (grayscale)
        """
        with open(preference_file, 'r') as f:
            self.preferences = json.load(f)
        
        self.image_folder = image_folder
        self.transform = transform
        self.color_mode = color_mode
        
        # Filter out skipped preferences (-1)
        self.preferences = [p for p in self.preferences if p['preference'] != -1]
        
        print(f"Loaded {len(self.preferences)} preference pairs")
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        # For this implementation, we'll generate crops on-demand
        # In practice, you might want to pre-generate and save crops
        try:
            # Create dummy images for demonstration
            # In real implementation, load the original image and crop
            crop_a = self._create_crop_image(pref['crop_a'])
            crop_b = self._create_crop_image(pref['crop_b'])
            
            if self.transform:
                crop_a = self.transform(crop_a)
                crop_b = self.transform(crop_b)
            
            # Create label (0 if A preferred, 1 if B preferred)
            label = torch.tensor(pref['preference'], dtype=torch.long)
            
            return crop_a, crop_b, label
            
        except Exception as e:
            print(f"Error loading preference {idx}: {e}")
            # Return dummy data if error
            dummy_size = (224, 224) if self.color_mode == 'RGB' else (224, 224)
            channels = 3 if self.color_mode == 'RGB' else 1
            dummy_tensor = torch.zeros(channels, *dummy_size)
            return dummy_tensor, dummy_tensor, torch.tensor(0, dtype=torch.long)
    
    def _create_crop_image(self, crop_info):
        """Create crop image from coordinates - placeholder implementation"""
        # This is a placeholder - in real implementation, you would:
        # 1. Load the original image
        # 2. Extract the crop using crop_info['coordinates']
        # 3. Resize to the expected size
        
        coords = crop_info['coordinates']  # (x, y, w, h)
        crop_size = (coords[2], coords[3])  # (width, height)
        
        # Create dummy image for demonstration
        if self.color_mode == 'RGB':
            dummy_img = Image.new('RGB', crop_size, color=(128, 128, 128))
        else:
            dummy_img = Image.new('L', crop_size, color=128)
            
        return dummy_img

class AestheticFeatureExtractor(nn.Module):
    """ResNet-based feature extractor fine-tuned for aesthetic preferences"""
    
    def __init__(self, model_name='resnet50', color_mode='RGB', embedding_dim=512, pretrained=True):
        super().__init__()
        
        self.color_mode = color_mode
        self.embedding_dim = embedding_dim
        
        # Load pre-trained ResNet
        if model_name == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Modify first layer for grayscale if needed
        if color_mode == 'L':
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Average the weights across RGB channels for grayscale
                self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Add normalization
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """Extract features from input image"""
        features = self.backbone(x)
        embedding = self.embedding(features)
        embedding = self.norm(embedding)
        return embedding

class PreferenceComparator(nn.Module):
    """Model that compares two image embeddings for preference learning"""
    
    def __init__(self, feature_extractor, comparison_dim=256):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.embedding_dim = feature_extractor.embedding_dim
        
        # Comparison network
        self.comparator = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, comparison_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(comparison_dim, comparison_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(comparison_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img_a, img_b):
        """
        Compare two images and output preference probability
        Returns: probability that img_b is preferred over img_a
        """
        embedding_a = self.feature_extractor(img_a)
        embedding_b = self.feature_extractor(img_b)
        
        # Concatenate embeddings
        combined = torch.cat([embedding_a, embedding_b], dim=1)
        
        # Predict preference
        preference_prob = self.comparator(combined)
        
        return preference_prob.squeeze(), embedding_a, embedding_b

class AestheticTrainer:
    """Trainer class for fine-tuning the aesthetic feature extractor"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (img_a, img_b, labels) in enumerate(pbar):
            img_a, img_b, labels = img_a.to(self.device), img_b.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_probs, emb_a, emb_b = self.model(img_a, img_b)
            
            # Convert labels to float (0.0 if A preferred, 1.0 if B preferred)
            target_probs = labels.float()
            
            # Calculate loss
            loss = criterion(pred_probs, target_probs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (pred_probs > 0.5).float()
            correct += (predicted == target_probs).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img_a, img_b, labels in tqdm(dataloader, desc="Validation"):
                img_a, img_b, labels = img_a.to(self.device), img_b.to(self.device), labels.to(self.device)
                
                # Forward pass
                pred_probs, _, _ = self.model(img_a, img_b)
                
                # Convert labels to float
                target_probs = labels.float()
                
                # Calculate loss
                loss = criterion(pred_probs, target_probs)
                
                # Statistics
                total_loss += loss.item()
                predicted = (pred_probs > 0.5).float()
                correct += (predicted == target_probs).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=1e-4, save_path='aesthetic_model.pth'):
        """Full training loop"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
                print(f"Saved best model to {save_path}")
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Training curves saved to {save_path}")

def get_data_transforms(color_mode='RGB', image_size=224):
    """Get data transformations for training and validation"""
    
    if color_mode == 'RGB':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
    else:  # Grayscale
        normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05) if color_mode == 'RGB' else transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Aesthetic Feature Extraction Training")
    
    # Data arguments
    parser.add_argument('--preference-file', type=str, required=True,
                       help='JSON file with preference data from Part 1')
    parser.add_argument('--image-folder', type=str, default=None,
                       help='Folder containing original images')
    parser.add_argument('--color-mode', type=str, choices=['RGB', 'L'], default='RGB',
                       help='Color mode: RGB or L (grayscale)')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, choices=['resnet18', 'resnet50', 'resnet101'], 
                       default='resnet50', help='ResNet model variant')
    parser.add_argument('--embedding-dim', type=int, default=512,
                       help='Dimension of output embeddings')
    parser.add_argument('--comparison-dim', type=int, default=256,
                       help='Dimension of comparison network')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training')
    
    # Output arguments
    parser.add_argument('--save-path', type=str, default='aesthetic_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--plot-curves', action='store_true',
                       help='Plot and save training curves')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, or cpu')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Color mode: {args.color_mode}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(args.color_mode, args.image_size)
    
    # Load dataset
    print("Loading preference dataset...")
    full_dataset = PreferenceDataset(
        args.preference_file, 
        args.image_folder,
        transform=train_transform,
        color_mode=args.color_mode
    )
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating {args.model_name} feature extractor...")
    feature_extractor = AestheticFeatureExtractor(
        model_name=args.model_name,
        color_mode=args.color_mode,
        embedding_dim=args.embedding_dim,
        pretrained=True
    )
    
    model = PreferenceComparator(feature_extractor, args.comparison_dim)
    
    # Create trainer and train
    trainer = AestheticTrainer(model, device)
    
    print("Starting training...")
    trainer.train(
        train_loader, 
        val_loader, 
        args.num_epochs, 
        args.learning_rate,
        args.save_path
    )
    
    print("Training completed!")
    
    # Plot curves if requested
    if args.plot_curves:
        trainer.plot_training_curves()
    
    # Save feature extractor separately for Part 3
    feature_extractor_path = args.save_path.replace('.pth', '_feature_extractor.pth')
    torch.save(feature_extractor.state_dict(), feature_extractor_path)
    print(f"Feature extractor saved to {feature_extractor_path}")

if __name__ == "__main__":
    main()