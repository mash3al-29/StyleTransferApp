import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from models.cnn_model import VGGFeatures, gram_matrix, content_loss, style_loss, StyleIntensityModule

# 1. Extract content features from VGG
# 2. Extract style features from both VGG and ViT
# 3. Combine the style features with appropriate weighting
# 4. Calculate content and style losses
# 5. Generate output using decoder
# 6. Fine-tune with style threshold control 


class TransformerFeatures(nn.Module):
    def __init__(self):
        super(TransformerFeatures, self).__init__()
        try:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)
            
            print("Successfully initialized Vision Transformer")
        except Exception as e:
            print(f"Error initializing Vision Transformer: {e}")
    
    def vit_features(self, x):
        B = x.shape[0]
        
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
            
        x = self.vit.patch_embed(x)
        
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        
        features = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in [2, 5, 8, 11]:
                features.append(x)
        
        return x[:, 0], features
    
    def reshape_transformer_features(self, features, content_shape):
        B, _, _ = features[0].shape
        _, C, H, W = content_shape
        
        reshaped_features = []
        for feat in features:
            patch_feat = feat[:, 1:, :]
            
            seq_len = patch_feat.shape[1]
            feat_h = int((seq_len)**0.5)
            feat_w = feat_h
            
            patch_feat = patch_feat.reshape(B, feat_h, feat_w, -1)
            
            patch_feat = patch_feat.permute(0, 3, 1, 2)
            
            patch_feat = F.interpolate(patch_feat, size=(H, W), mode='bilinear', align_corners=True)
            
            reshaped_features.append(patch_feat)
        
        return reshaped_features
    
    def forward(self, x):
        vgg = VGGFeatures()
        content_features, _ = vgg(x)
        
        _, vit_features = self.vit_features(x)
        
        vit_style_features = self.reshape_transformer_features(vit_features, content_features.shape)
        
        return content_features, vit_style_features

class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        self.vgg = VGGFeatures()
        
        self.vit = TransformerFeatures()
        
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
    
    def reshape_transformer_features(self, features, content_shape):
        B, _, _ = features[0].shape
        _, C, H, W = content_shape
        
        reshaped_features = []
        for feat in features:
            patch_feat = feat[:, 1:, :]
            
            seq_len = patch_feat.shape[1]
            feat_h = int((seq_len)**0.5)
            feat_w = feat_h
            
            patch_feat = patch_feat.reshape(B, feat_h, feat_w, -1)
            
            patch_feat = patch_feat.permute(0, 3, 1, 2)
            
            patch_feat = F.interpolate(patch_feat, size=(H, W), mode='bilinear', align_corners=True)
            
            reshaped_features.append(patch_feat)
        
        return reshaped_features
    
    def extract_vit_features(self, x):
        return self.vit.vit_features(x)
    
    def forward(self, content_img, style_img, style_threshold):
        content_features, _ = self.vgg(content_img)
        
        modulated_features = self.style_intensity(content_features, style_threshold)
        
        output = self.decoder(modulated_features)
        
        return output