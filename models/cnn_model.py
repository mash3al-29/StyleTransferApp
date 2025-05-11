import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
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