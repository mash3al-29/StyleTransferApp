!pip install -q ipywidgets
!pip install -q tqdm

import os
import numpy as np
from pathlib import Path
import random
from PIL import Image
import logging
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Uses VGG19 layers (conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 for style; conv4_2 for content)
# 2. Computes Gram matrices for style representation
# 3. Uses content and style losses with appropriate weighting using alpha and beta
# 4. Allows control of style transfer strength through style_threshold parameter

class StyleTransferDataset(Dataset):
    def __init__(self, 
                 content_dir: str,
                 style_dir: str,
                 transform=None,
                 style_threshold: float = 0.5):
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.style_threshold = style_threshold
        
        self.content_images = sorted([f for f in self.content_dir.glob('*.jpg')])
        self.style_images = sorted([f for f in self.style_dir.glob('*.jpg')])
        
        if not self.content_images:
            raise ValueError(f"No content images found in {self.content_dir}")
        if not self.style_images:
            raise ValueError(f"No style images found in {self.style_dir}")
            
        logger.info(f"Loaded {len(self.content_images)} content images and {len(self.style_images)} style images")
    
    def __len__(self) -> int:
        return len(self.content_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        content_path = self.content_images[idx]
        style_path = random.choice(self.style_images)
        
        try:
            content_img = Image.open(content_path).convert('RGB')
            style_img = Image.open(style_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            content_img = Image.new('RGB', (256, 256), 'black')
            style_img = Image.new('RGB', (256, 256), 'black')
        
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        
        style_threshold = torch.tensor(self.style_threshold, dtype=torch.float32)
        return content_img, style_img, style_threshold

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])
        self.slice5 = nn.Sequential(*list(vgg.children())[21:22])
        self.slice6 = nn.Sequential(*list(vgg.children())[22:30])
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
    
    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        h6 = self.slice6(h5)
        
        content_features = h5
        
        style_features = [h1, h2, h3, h4, h6]
        
        return content_features, style_features

class StyleIntensityModule(nn.Module):
    def __init__(self):
        super(StyleIntensityModule, self).__init__()
        self.style_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.style_norm1 = nn.InstanceNorm2d(512)
        self.style_conv2 = nn.Conv2d(512, 512, kernel_size=1)
        
        nn.init.kaiming_normal_(self.style_conv1.weight)
        nn.init.kaiming_normal_(self.style_conv2.weight)
        nn.init.constant_(self.style_conv1.bias, 0)
        nn.init.constant_(self.style_conv2.bias, 0)
    
    def forward(self, content_features, style_threshold):
        b = content_features.size(0)
        style_threshold = style_threshold.view(b, 1, 1, 1)
        
        content_only = content_features * 1.0
        
        style_effect = self.style_conv1(content_features)
        style_effect = F.relu(self.style_norm1(style_effect))
        style_effect = self.style_conv2(style_effect)
        style_max = content_features + style_effect * 2.0
        
        low_threshold_mask = torch.sigmoid((0.3 - style_threshold) * 10)
        content_blur = F.avg_pool2d(content_features, kernel_size=3, stride=1, padding=1)
        content_sharpened = content_features + (content_features - content_blur) * 0.5
        content_only = content_only * (1 - low_threshold_mask) + content_sharpened * low_threshold_mask
        
        high_threshold_mask = torch.sigmoid((style_threshold - 0.7) * 10)
        channel_attention = F.adaptive_avg_pool2d(style_max, 1)
        channel_attention = torch.sigmoid(channel_attention) * 1.5
        style_enhanced = style_max * channel_attention
        style_max = style_max * (1 - high_threshold_mask) + style_enhanced * high_threshold_mask
        
        output = content_only * (1 - style_threshold) + style_max * style_threshold
        
        return output

class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        self.vgg = VGGFeatures()
        
        self.style_intensity = StyleIntensityModule()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, content_img, style_img, style_threshold):
        content_features, _ = self.vgg(content_img)
        modulated_features = self.style_intensity(content_features, style_threshold)
        output = self.decoder(modulated_features)
        return output

# Vision Transformer feature extractor for style transfer
class TransformerFeatures(nn.Module):
    def __init__(self):
        super(TransformerFeatures, self).__init__()
        try:
            import timm
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)
            
            self.vgg = VGGFeatures()
            
            print("Successfully initialized Vision Transformer")
        except ImportError:
            print("Warning: timm package not found. Install with: !pip install -q timm")
            self.vit = None
    
    def extract_vit_features(self, x):
        if self.vit is None:
            return None, None
        
        B = x.shape[0]
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
            
        x = self.vit.patch_embed(x)
        
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        # Extract features from different transformer blocks
        features = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in [2, 5, 8, 11]:  # Extract from different depths
                features.append(x)
        
        return x[:, 0], features
    
    def reshape_transformer_features(self, features, content_shape):
        # For each feature in the list, reshape from [B, N, C] to [B, C, H, W]
        B, _, _ = features[0].shape
        _, C, H, W = content_shape
        
        reshaped_features = []
        for feat in features:
            patch_feat = feat[:, 1:, :]
            
            seq_len = patch_feat.shape[1]
            feat_h = int((seq_len)**0.5)
            feat_w = feat_h
            
            # Reshape to [B, H, W, C]
            patch_feat = patch_feat.reshape(B, feat_h, feat_w, -1)
            
            # Permute to [B, C, H, W]
            patch_feat = patch_feat.permute(0, 3, 1, 2)
            
            patch_feat = F.interpolate(patch_feat, size=(H, W), mode='bilinear', align_corners=True)
            
            reshaped_features.append(patch_feat)
        
        return reshaped_features
    
    def forward(self, x):
        content_features, _ = self.vgg(x)
        
        if self.vit is not None:
            _, vit_features = self.extract_vit_features(x)
            
            # Reshape ViT features to match VGG feature dimensions
            vit_style_features = self.reshape_transformer_features(vit_features, content_features.shape)
        else:
            _, vit_style_features = self.vgg(x)
        
        return content_features, vit_style_features

class ViTStyleTransferModel(nn.Module):
    def __init__(self):
        super(ViTStyleTransferModel, self).__init__()
        self.vgg = VGGFeatures()
        self.vit = TransformerFeatures()
        
        self.style_intensity = StyleIntensityModule()
        
        # Decoder architecture (same as CNN model for compatibility)
        self.decoder = nn.Sequential(
            # Start from content features [B, 512, 32, 32]
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # [B, 256, 64, 64]
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # [B, 128, 128, 128]
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # [B, 64, 256, 256]
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, content_img, style_img, style_threshold):
        content_features, _ = self.vgg(content_img)
        modulated_features = self.style_intensity(content_features, style_threshold)
        output = self.decoder(modulated_features)
        return output

def gram_matrix(x):
    b, c, h, w = x.size()
    # Reshape to (batch_size, channels, height*width)
    features = x.view(b, c, h * w)
    # Compute gram matrix: G = FF^T / (channels * height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    # Normalize by the number of elements in each feature map
    return gram.div(c * h * w)

def content_loss(content_features, target_features):
    return F.mse_loss(content_features, target_features)

def style_loss(output_style_features, target_style_features):
    loss = 0
    style_weights = [1.0/len(output_style_features)] * len(output_style_features)
    
    for i, (sf, tf) in enumerate(zip(output_style_features, target_style_features)):
        gram_s = gram_matrix(sf)
        gram_t = gram_matrix(tf)
        layer_loss = F.mse_loss(gram_s, gram_t) * style_weights[i]
        loss += layer_loss
    
    return loss

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, model_type='cnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nTraining {model_type.upper()} model on device: {device}")
    print(f"Number of batches per epoch: {len(train_loader)}\n")
    
    # Use Adam optimizer with a lower learning rate for better convergence instead of L-BFGS since L-BFGS is very time consuming compared to our use case
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    CONTENT_WEIGHT = 1.0    # Content weight
    STYLE_WEIGHT = 1000.0   # Style weight
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    model_filename = f"best_{model_type}_style_transfer_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_content_loss = 0
        train_style_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (content_img, style_img, style_threshold) in enumerate(progress_bar):
            try:
                # Move data to device
                content_img = content_img.to(device)
                style_img = style_img.to(device)
                style_threshold = style_threshold.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                output = model(content_img, style_img, style_threshold)
                
                with torch.no_grad():
                    if model_type == 'vit':
                        content_features, _ = model.vgg(content_img)  # Content features
                        _, style_features = model.vit(style_img)      # Style features from ViT
                    else:
                        content_features, _ = model.vgg(content_img)  # Content features 
                        _, style_features = model.vgg(style_img)      # Style features from VGG
                
                if model_type == 'vit':
                    output_content, _ = model.vgg(output)  # Content features
                    _, output_style = model.vit(output)    # Style features from ViT
                else: 
                    output_content, output_style = model.vgg(output)
                
                # Apply style weight based on style_threshold
                batch_style_weight = STYLE_WEIGHT * style_threshold.mean().item()
                
                # Calculate losses with weights
                c_loss = content_loss(output_content, content_features) * CONTENT_WEIGHT
                
                # Style loss is weighted by beta
                s_loss = style_loss(output_style, style_features) * batch_style_weight
                
                total_loss = c_loss + s_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_content_loss += c_loss.item()
                train_style_loss += s_loss.item()
                batch_count += 1
                
                progress_bar.set_postfix({
                    'loss': f'{total_loss.item():.6f}',
                    'content': f'{c_loss.item():.6f}',
                    'style': f'{s_loss.item():.6f}'
                })
                
                if batch_idx % 5 == 0:
                    progress_bar.write(
                        f"Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {total_loss.item():.6f} | "
                        f"Content: {c_loss.item():.6f} | "
                        f"Style: {s_loss.item():.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                
                if (epoch * len(train_loader) + batch_idx) % 500 == 0 and batch_idx > 0:
                    # Save a sample image
                    with torch.no_grad():
                        denorm = lambda x: x * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device) + \
                                      torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
                        
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(131)
                        img = denorm(content_img[0]).cpu().clamp(0, 1)
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title('Content Image')
                        plt.axis('off')
                        
                        plt.subplot(132)
                        img = denorm(style_img[0]).cpu().clamp(0, 1)
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title('Style Image')
                        plt.axis('off')
                        
                        plt.subplot(133)
                        img = denorm(output[0]).cpu().clamp(0, 1)
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title('Generated Image')
                        plt.axis('off')
                        
                        plt.savefig(f'intermediate_result_epoch_{epoch+1}_batch_{batch_idx}.png')
                        plt.close()
                
            except Exception as e:
                print(f"Error in training batch {batch_count}: {str(e)}")
                continue
        
        train_loss /= batch_count
        train_content_loss /= batch_count
        train_style_loss /= batch_count
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        val_content_loss = 0
        val_style_loss = 0
        val_count = 0
        
        print("\nValidation:")
        with torch.no_grad():
            for content_img, style_img, style_threshold in val_loader:
                if val_count >= 10:  # Only validate on a few batches
                    break
                    
                content_img = content_img.to(device)
                style_img = style_img.to(device)
                style_threshold = style_threshold.to(device)
                
                output = model(content_img, style_img, style_threshold)
                content_features, _ = model.vgg(content_img)
                _, style_features = model.vgg(style_img)
                output_content, output_style = model.vgg(output)
                
                # Apply style weight based on style_threshold for validation
                batch_style_weight = STYLE_WEIGHT * style_threshold.mean().item()
                
                # Calculate losses
                c_loss = content_loss(output_content, content_features) * CONTENT_WEIGHT
                s_loss = style_loss(output_style, style_features) * batch_style_weight
                total_loss = c_loss + s_loss
                
                val_loss += total_loss.item()
                val_content_loss += c_loss.item()
                val_style_loss += s_loss.item()
                val_count += 1
        
        val_loss /= max(val_count, 1)  # Avoid division by zero
        val_content_loss /= max(val_count, 1)
        val_style_loss /= max(val_count, 1)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Train Loss: {train_loss:.6f}")
        print(f"Average Content Loss: {train_content_loss:.6f}")
        print(f"Average Style Loss: {train_style_loss:.6f}")
        
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Validation Content Loss: {val_content_loss:.6f}")
        print(f"Validation Style Loss: {val_style_loss:.6f}")
        print("-" * 50)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_filename)
            print(f"Saved best {model_type.upper()} model with validation loss: {val_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_type}_style_transfer_model_epoch_{epoch+1}.pth')
    
    return train_losses, val_losses

def process_content_images(source_dir, target_dir, count):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    categories = ['architecure', 'art and culture', 'food and drinks', 'travel and adventure']
    all_images = []
    
    for category in categories:
        category_path = source_dir / category
        if category_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                all_images.extend(list(category_path.glob(ext)))
                all_images.extend(list(category_path.glob(ext.upper())))
    
    logger.info(f"Found {len(all_images)} content images across all categories")
    
    if not all_images:
        raise ValueError(f"No content images found in {source_dir}")
    
    count = min(count, 2000)
    selected_images = random.sample(all_images, min(count, len(all_images)))
    
    for idx, img_path in enumerate(tqdm(selected_images, desc='Processing content images')):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Determine split - adjusted split ratios
            if idx < len(selected_images) * 0.8:
                split = 'train'
            elif idx < len(selected_images) * 0.9:
                split = 'val'
            else:
                split = 'test'
            
            save_path = target_dir / split / f'content_{idx:04d}.jpg'
            img.save(save_path, 'JPEG', quality=95)
            
        except Exception as e:
            logger.error(f'Error processing content image {img_path}: {e}')
            continue

def process_style_images(source_dir, target_dir, count):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    for split in ['train', 'val', 'test']:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    all_images = []
    for ext in ['*.jpg', '*.jpeg']:
        all_images.extend(list(source_dir.glob(ext)))
    
    logger.info(f"Found {len(all_images)} style images")
    
    if not all_images:
        raise ValueError(f"No style images found in {source_dir}")
    

    count = min(count, 2000)
    selected_images = random.sample(all_images, min(count, len(all_images)))
    
    for idx, img_path in enumerate(tqdm(selected_images, desc='Processing style images')):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            if idx < len(selected_images) * 0.8:
                split = 'train'
            elif idx < len(selected_images) * 0.9:
                split = 'val'
            else:
                split = 'test'
            
            save_path = target_dir / split / f'style_{idx:04d}.jpg'
            img.save(save_path, 'JPEG', quality=95)
            
        except Exception as e:
            logger.error(f'Error processing style image {img_path}: {e}')
            continue

def main():
    content_input_dir = Path('/kaggle/input/image-classification/images/images')
    # /kaggle/input/wikiart/Pointillism
    style_input_dir = Path('/kaggle/input/wikiart/New_Realism')
    base_dir = Path('/kaggle/working/data')
    
    logger.info(f"Content input directory: {content_input_dir}")
    logger.info(f"Style input directory: {style_input_dir}")
    logger.info(f"Working directory: {base_dir}")
    
    if not content_input_dir.exists():
        raise ValueError(f"Content input directory not found: {content_input_dir}")
    if not style_input_dir.exists():
        raise ValueError(f"Style input directory not found: {style_input_dir}")
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    content_dir = base_dir / 'content'
    logger.info("Processing content images...")
    process_content_images(content_input_dir, content_dir, 2000)  # Increased for more patches
    
    style_dir = base_dir / 'style'
    logger.info("Processing style images...")
    process_style_images(style_input_dir, style_dir, 400)  # Increased for more style variety
    
    train_content_dir = content_dir / 'train'
    train_style_dir = style_dir / 'train'
    
    if not train_content_dir.exists() or not any(train_content_dir.glob('*.jpg')):
        raise ValueError(f"No processed content images found in {train_content_dir}")
    if not train_style_dir.exists() or not any(train_style_dir.glob('*.jpg')):
        raise ValueError(f"No processed style images found in {train_style_dir}")
    
    logger.info("Creating datasets...")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = StyleTransferDataset(
        content_dir / 'train',
        style_dir / 'train',
        transform=transform
    )
    
    val_dataset = StyleTransferDataset(
        content_dir / 'val',
        style_dir / 'val',
        transform=transform
    )
    
    test_dataset = StyleTransferDataset(
        content_dir / 'test',
        style_dir / 'test',
        transform=transform
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=12,
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=12, 
        num_workers=2,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=12, 
        num_workers=2,
        drop_last=True
    )
    
    logger.info(f"Number of iterations per epoch: {len(train_loader)}")
    
    # Create and train CNN model
    logger.info("Creating and training CNN model...")
    cnn_model = StyleTransferModel()
    cnn_train_losses, cnn_val_losses = train_model(cnn_model, train_loader, val_loader, model_type='cnn')
    
    plt.figure(figsize=(10, 5))
    plt.plot(cnn_train_losses, label='Training Loss')
    plt.plot(cnn_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Model Training and Validation Losses')
    plt.legend()
    plt.savefig('cnn_training_curves.png')
    plt.close()
    
    # Create and train ViT model
    try:
        logger.info("Creating and training Vision Transformer model...")
        vit_model = ViTStyleTransferModel()
        
        if vit_model.vit.vit is not None:
            vit_train_losses, vit_val_losses = train_model(vit_model, train_loader, val_loader, model_type='vit')
            
            # Plot ViT training curves
            plt.figure(figsize=(10, 5))
            plt.plot(vit_train_losses, label='Training Loss')
            plt.plot(vit_val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('ViT Model Training and Validation Losses')
            plt.legend()
            plt.savefig('vit_training_curves.png')
            plt.close()
            
            # Save ViT model
            torch.save(vit_model.state_dict(), 'vit_style_transfer_model.pth')
        else:
            logger.info("Vision Transformer not available, skipping ViT model training")
    except Exception as e:
        logger.error(f"Error training Vision Transformer model: {e}")
        logger.info("Continuing with CNN model only")
    
    logger.info("Generating test samples...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CNN model
    cnn_model.eval()
    cnn_model = cnn_model.to(device)
    
    try:
        vit_model.eval()
        vit_model = vit_model.to(device)
        vit_available = True
    except:
        vit_available = False
    
    # Denormalization function
    denorm = lambda x: x * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device) + \
                      torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
    
    with torch.no_grad():
        for i, (content_img, style_img, _) in enumerate(test_loader):
            if i >= 3:
                break
            
            content_img = content_img.to(device)
            style_img = style_img.to(device)
            
            # Test with different style thresholds
            for threshold in [0.1, 0.5, 1.0]:
                style_threshold = torch.tensor([threshold] * content_img.size(0)).to(device)
                
                cnn_output = cnn_model(content_img, style_img, style_threshold)
                
                if vit_available:
                    vit_output = vit_model(content_img, style_img, style_threshold)
                
                for j in range(min(2, content_img.size(0))):
                    if vit_available:
                        plt.figure(figsize=(20, 5))
                    else:
                        plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 4 if vit_available else 3, 1)
                    img = denorm(content_img[j]).cpu().clamp(0, 1)
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title('Content Image')
                    plt.axis('off')
                    
                    plt.subplot(1, 4 if vit_available else 3, 2)
                    img = denorm(style_img[j]).cpu().clamp(0, 1)
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title('Style Image')
                    plt.axis('off')
                    
                    plt.subplot(1, 4 if vit_available else 3, 3)
                    img = denorm(cnn_output[j]).cpu().clamp(0, 1)
                    plt.imshow(img.permute(1, 2, 0))
                    plt.title(f'CNN Output (Threshold={threshold})')
                    plt.axis('off')
                    
                    if vit_available:
                        plt.subplot(1, 4, 4)
                        img = denorm(vit_output[j]).cpu().clamp(0, 1)
                        plt.imshow(img.permute(1, 2, 0))
                        plt.title(f'ViT Output (Threshold={threshold})')
                        plt.axis('off')
                    
                    plt.savefig(f'result_{i}_{j}_threshold_{threshold}.png')
                    plt.close()
    
    logger.info("Processing completed successfully!")
    
    print("\n" + "="*50)
    print("TRAINED MODELS SUMMARY")
    print("="*50)
    print("1. CNN Model: best_cnn_style_transfer_model.pth")
    if vit_available:
        print("2. ViT Model: best_vit_style_transfer_model.pth")
    print("\nBoth models support style intensity control via the style_threshold parameter")
    print("Style threshold ranges from 0.1 (minimal style) to 1.0 (maximum style)")
    print("="*50)

if __name__ == "__main__":
    try:
        import timm
    except ImportError:
        print("Installing timm package for Vision Transformer support...")
        !pip install -q timm
    
    main()