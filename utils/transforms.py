import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(image, target_size=(256, 256)):
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)

def postprocess_image(tensor):
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    tensor = torch.clamp(tensor, 0, 1)
    
    tensor = tensor.permute(1, 2, 0).numpy() * 255
    return Image.fromarray(tensor.astype(np.uint8))

def make_grid(images, rows=1, cols=None):
    if not images:
        return None
    
    if cols is None:
        cols = len(images) // rows
        if len(images) % rows:
            cols += 1
    
    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * width, row * height))
    
    return grid 