import os
import torch
import numpy as np
from PIL import Image
import io
import base64
from flask import Flask, render_template, request, jsonify
from models.cnn_model import StyleTransferModel as CNNStyleTransferModel
from models.cnn_model import VGGFeatures, gram_matrix, content_loss, style_loss
from models.vit_model import StyleTransferModel as ViTStyleTransferModel
from utils.transforms import preprocess_image, postprocess_image
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/dummy', exist_ok=True)

dummy_style_path = os.path.join('static/dummy', 'dummy-style.jpg')
if not os.path.exists(dummy_style_path):
    try:
        dummy_style = Image.new('RGB', (256, 256), (128, 128, 128))
        for x in range(256):
            for y in range(256):
                if (x + y) % 20 < 10:
                    value = 128 + ((x * y) % 100)
                    dummy_style.putpixel((x, y), (value, value, value))
        dummy_style.save(dummy_style_path)
        print(f"Created neutral dummy style image at {dummy_style_path}")
    except Exception as e:
        print(f"Error creating dummy style image: {e}")

cnn_model = None
try:
    if os.path.exists('kaggle_notebook_outputs/best_cnn_style_transfer_model.pth'):
        print("Loading Kaggle pre-trained CNN model with New Realism style...")
        cnn_model = CNNStyleTransferModel()
        cnn_model.load_state_dict(torch.load('kaggle_notebook_outputs/best_cnn_style_transfer_model.pth', 
                                         map_location=torch.device('cpu')))
        cnn_model.eval()
        print("CNN Model loaded successfully!")
    else:
        print("ERROR: Kaggle CNN model not found. Please ensure kaggle_notebook_outputs/best_cnn_style_transfer_model.pth exists.")
except Exception as e:
    print(f"Error loading CNN model: {e}")

vit_model = None

def get_model(model_type):
    global vit_model
    if model_type == 'vit':
        try:
            if vit_model is None:
                from models.vit_model import StyleTransferModel as ViTStyleTransferModel
                
                print("Creating new ViT model instance...")
                vit_model = ViTStyleTransferModel()
                
                if os.path.exists('kaggle_notebook_outputs/best_vit_style_transfer_model.pth'):
                    print("Loading Kaggle pre-trained ViT model with New Realism style...")
                    
                    state_dict = torch.load('kaggle_notebook_outputs/best_vit_style_transfer_model.pth', 
                                         map_location=torch.device('cpu'))
                    
                    print(f"Model keys: {', '.join(list(vit_model.state_dict().keys())[:5])}...")
                    print(f"Loaded state dict keys: {', '.join(list(state_dict.keys())[:5])}...")
                    
                    try:
                        vit_model.load_state_dict(state_dict, strict=False)
                        vit_model.eval()
                        print("ViT Model initialized successfully!")
                    except Exception as e:
                        print(f"Error loading ViT model state dict: {e}")
                        print("Will continue with initialized (untrained) ViT model.")
                else:
                    print("Kaggle ViT model not found. Falling back to CNN model.")
                    return cnn_model
            return vit_model
        except Exception as e:
            print(f"Error initializing ViT model: {e}")
            return cnn_model
    
    return cnn_model if model_type == 'cnn' or cnn_model is not None else None

def color_transfer(source, target):
    try:
        source = np.array(source).astype(np.float32) / 255.0
        target = np.array(target).astype(np.float32) / 255.0
        
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        
        source_mean, source_std = [], []
        for i in range(3):
            source_mean.append(np.mean(source_lab[:,:,i]))
            source_std.append(np.std(source_lab[:,:,i]))
        
        target_mean, target_std = [], []
        for i in range(3):
            target_mean.append(np.mean(target_lab[:,:,i]))
            target_std.append(np.std(target_lab[:,:,i]))
        
        result_lab = np.copy(target_lab)
        for i in range(3):
            result_lab[:,:,i] = ((result_lab[:,:,i] - target_mean[i]) * 
                               (source_std[i] / max(target_std[i], 1e-5))) + source_mean[i]
        
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        result = np.clip(result, 0, 1) * 255
        return Image.fromarray(result.astype(np.uint8))
    except Exception as e:
        print(f"Error in color transfer: {str(e)}")
        return target

def neural_style_transfer(content_img, style_img, style_weight=1e6, content_weight=1, num_steps=200, model_type='cnn', style_threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    content_img = content_img.to(device)
    style_img = style_img.to(device)
    
    try:
        if model_type == 'vit':
            try:
                from models.vit_model import TransformerFeatures, VGGFeatures
                
                vit_feature_extractor = TransformerFeatures().to(device)
                vgg_feature_extractor = VGGFeatures().to(device)
                
                with torch.no_grad():
                    content_features, _ = vgg_feature_extractor(content_img)
                    _, style_features = vit_feature_extractor.vit_features(style_img)
                    
                    style_features = vit_feature_extractor.reshape_transformer_features(
                        style_features, content_features.shape
                    )
                    
                    style_grams = [gram_matrix(sf) for sf in style_features]
            except Exception as e:
                print(f"Error initializing ViT for direct transfer: {e}")
                print("Falling back to CNN model")
                model_type = 'cnn'
                feature_extractor = VGGFeatures().to(device)
                
                with torch.no_grad():
                    content_features, _ = feature_extractor(content_img)
                    _, style_features = feature_extractor(style_img)
                    
                    style_grams = [gram_matrix(sf) for sf in style_features]
        else:
            feature_extractor = VGGFeatures().to(device)
            
            with torch.no_grad():
                content_features, _ = feature_extractor(content_img)
                _, style_features = feature_extractor(style_img)
                
                style_grams = [gram_matrix(sf) for sf in style_features]
        
        if style_threshold > 0.6:
            input_img = style_img.clone().detach()
            noise = torch.randn_like(input_img) * 0.1
            input_img = torch.clamp(input_img + noise, 0, 1)
        else:
            input_img = content_img.clone().detach()
            noise = torch.randn_like(input_img) * 0.03
            input_img = torch.clamp(input_img + noise, 0, 1)
        
        input_img.requires_grad_(True)
        
        optimizer = optim.LBFGS([input_img], lr=0.1)
        
        adjusted_steps = min(int(num_steps * (style_weight / 1e6)), 300)
        adjusted_steps = max(adjusted_steps, 50)
        
        if style_threshold > 0.6:
            style_weights = [5.0, 3.0, 1.0, 0.5, 0.5]
        else:
            style_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        style_weights = [w/sum(style_weights) for w in style_weights]
        
        step = 0
        best_img = None
        best_loss = float('inf')
        
        while step < adjusted_steps:
            def closure():
                nonlocal step, best_img, best_loss
                
                try:
                    optimizer.zero_grad()
                    
                    if model_type == 'vit':
                        try:
                            out_content, _ = vgg_feature_extractor(input_img)
                            _, out_style_raw = vit_feature_extractor.vit_features(input_img)
                            
                            out_style = vit_feature_extractor.reshape_transformer_features(
                                out_style_raw, out_content.shape
                            )
                        except Exception as e:
                            print(f"Error in ViT feature extraction during optimization: {e}")
                            return best_loss
                    else:
                        out_content, out_style = feature_extractor(input_img)
                    
                    c_loss = content_loss(out_content, content_features) * content_weight
                    
                    s_loss = 0
                    for i, (sf, target_gram) in enumerate(zip(out_style, style_grams)):
                        try:
                            layer_style_loss = F.mse_loss(gram_matrix(sf), target_gram) * style_weights[i]
                            s_loss += layer_style_loss
                        except Exception as e:
                            print(f"Error calculating style loss for layer {i}: {e}")
                    s_loss *= style_weight
                    
                    total_loss = c_loss + s_loss
                    
                    if total_loss.item() < best_loss:
                        best_loss = total_loss.item()
                        with torch.no_grad():
                            best_img = input_img.clone().detach().cpu()
                    
                    total_loss.backward()
                    
                    step += 1
                    
                    if step % 20 == 0 or step == 1:
                        print(f"Step {step}/{adjusted_steps}, Content Loss: {c_loss.item():.4f}, Style Loss: {s_loss.item():.4f}")
                        
                    return total_loss
                except Exception as e:
                    print(f"Error in optimization step: {e}")
                    return torch.tensor(best_loss, device=device)
            
            optimizer.step(closure)
            
            with torch.no_grad():
                input_img.clamp_(0, 1)
            
            if step >= adjusted_steps:
                break
        
        return best_img if best_img is not None else input_img
    except Exception as e:
        print(f"Error in neural_style_transfer: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    style_threshold = float(request.form.get('style_threshold', 0.5))
    model_type = request.form.get('model_type', 'cnn')
    transfer_method = request.form.get('transfer_method', 'direct')
    
    style_required = transfer_method == 'direct'
    
    if 'content_image' not in request.files:
        return jsonify({'error': 'Content image is required'}), 400
    
    if style_required and 'style_image' not in request.files:
        return jsonify({'error': 'Style image is required for direct style transfer'}), 400
    
    try:
        content_file = request.files['content_image']
        
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
        content_file.save(content_path)
        
        content_img = Image.open(content_path).convert('RGB')
        
        original_content_size = content_img.size
        
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'default_new_realism_style.jpg')
        
        if style_required and 'style_image' in request.files:
            style_file = request.files['style_image']
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
            style_file.save(style_path)
        elif not os.path.exists(style_path):
            default_style = Image.new('RGB', (256, 256), (128, 128, 128))
            for x in range(256):
                for y in range(256):
                    if (x + y) % 20 < 10:
                        value = 128 + ((x * y) % 100)
                        default_style.putpixel((x, y), (value, value, value))
            default_style.save(style_path)
        
        style_img = Image.open(style_path).convert('RGB')
        
        max_size = 384
        
        content_size = content_img.size
        style_size = style_img.size
        
        if max(content_size) > max_size:
            ratio = max_size / max(content_size)
            new_size = (int(content_size[0] * ratio), int(content_size[1] * ratio))
            print(f"Resizing content image from {content_size} to {new_size} for processing")
            content_img = content_img.resize(new_size, Image.LANCZOS)
        
        if max(style_size) > max_size:
            ratio = max_size / max(style_size)
            new_size = (int(style_size[0] * ratio), int(style_size[1] * ratio))
            print(f"Resizing style image from {style_size} to {new_size} for processing")
            style_img = style_img.resize(new_size, Image.LANCZOS)
        
        content_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'content_resized.jpg'))
        style_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'style_resized.jpg'))
        
        content_tensor = preprocess_image(content_img)
        style_tensor = preprocess_image(style_img)
        
        content_tensor = content_tensor.unsqueeze(0)
        style_tensor = style_tensor.unsqueeze(0)
        
        if transfer_method == 'pretrained':
            print(f"Using pre-trained model with style threshold: {style_threshold}")
            
            model = get_model(model_type)
            
            if model is None:
                return jsonify({'error': 'Model not available'}), 500
            
            with torch.no_grad():
                style_threshold_tensor = torch.tensor([style_threshold], dtype=torch.float32)
                print(f"Applying New Realism style with threshold: {style_threshold}")
                output = model(content_tensor, style_tensor, style_threshold_tensor)
            
            stylized_img = postprocess_image(output[0])
            
        else:
            if style_threshold > 0.6:
                try:
                    print(f"Applying initial color transfer with strength proportional to threshold: {style_threshold}")
                    color_matched_content = color_transfer(style_img, content_img)
                    color_matched_content.save(os.path.join(app.config['UPLOAD_FOLDER'], 'content_color_matched.jpg'))
                    content_img = color_matched_content
                    content_tensor = preprocess_image(content_img).unsqueeze(0)
                except Exception as e:
                    print(f"Error in color enhancement: {str(e)}")
            
            print(f"Applying direct style transfer with threshold: {style_threshold}, model: {model_type}")
            
            style_weight = 1e6 * (style_threshold ** 1.5)
            
            base_steps = 100
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cpu':
                base_steps = 100
                
            num_steps = int(base_steps * (0.5 + style_threshold))
            
            try:
                output = neural_style_transfer(
                    content_tensor, 
                    style_tensor, 
                    style_weight=style_weight, 
                    num_steps=num_steps,
                    model_type=model_type,
                    style_threshold=style_threshold
                )
                
                if output is None:
                    if model_type == 'vit':
                        print("ViT model failed, falling back to CNN model")
                        model_type = 'cnn'
                        output = neural_style_transfer(
                            content_tensor, 
                            style_tensor, 
                            style_weight=style_weight, 
                            num_steps=num_steps,
                            model_type='cnn',
                            style_threshold=style_threshold
                        )
                    
                    if output is None:
                        raise RuntimeError("Failed to generate output")
                
                stylized_img = postprocess_image(output[0])
                
                if style_threshold > 0.8 and transfer_method != 'pretrained':
                    try:
                        print(f"Applying final color enhancement with threshold: {style_threshold}")
                        stylized_img = color_transfer(style_img, stylized_img)
                    except Exception as e:
                        print(f"Error in color enhancement: {str(e)}")
                
                try:
                    stylized_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'result_before_resize.jpg'), quality=95)
                except Exception as e:
                    print(f"Error saving intermediate result: {e}")
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "cuda" in error_str:
                    max_size = max_size // 2
                    
                    ratio = max_size / max(content_img.size)
                    content_img = content_img.resize((int(content_img.size[0] * ratio), int(content_img.size[1] * ratio)), Image.LANCZOS)
                    
                    ratio = max_size / max(style_img.size)
                    style_img = style_img.resize((int(style_img.size[0] * ratio), int(style_img.size[1] * ratio)), Image.LANCZOS)
                    
                    content_tensor = preprocess_image(content_img).unsqueeze(0)
                    style_tensor = preprocess_image(style_img).unsqueeze(0)
                    
                    num_steps = max(30, num_steps // 2)
                    
                    output = neural_style_transfer(
                        content_tensor, 
                        style_tensor, 
                        style_weight=style_weight, 
                        num_steps=num_steps,
                        model_type=model_type,
                        style_threshold=style_threshold
                    )
                    
                    if output is None:
                        if model_type == 'vit':
                            print("ViT model failed, falling back to CNN model")
                            model_type = 'cnn'
                            output = neural_style_transfer(
                                content_tensor, 
                                style_tensor, 
                                style_weight=style_weight, 
                                num_steps=num_steps,
                                model_type='cnn',
                                style_threshold=style_threshold
                            )
                        
                        if output is None:
                            raise RuntimeError("Failed to generate output even with smaller images")
                    
                    stylized_img = postprocess_image(output[0])
                    
                    if style_threshold > 0.8 and transfer_method != 'pretrained':
                        try:
                            print(f"Applying final color enhancement with threshold: {style_threshold}")
                            stylized_img = color_transfer(style_img, stylized_img)
                        except Exception as e:
                            print(f"Error in color enhancement: {str(e)}")
                else:
                    raise
        
        if stylized_img.size != original_content_size and max(original_content_size) <= 1024:
            print(f"Resizing output back to original content size: {original_content_size}")
            stylized_img = stylized_img.resize(original_content_size, Image.LANCZOS)
        
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        stylized_img.save(result_path, quality=95)
        
        buffered = io.BytesIO()
        stylized_img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'result': f'data:image/jpeg;base64,{img_str}',
            'content_path': '/static/uploads/content.jpg',
            'style_path': '/static/uploads/style.jpg',
            'result_path': '/static/uploads/result.jpg',
            'style_threshold': style_threshold,
            'model_type': model_type,
            'transfer_method': transfer_method
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('kaggle_notebook_outputs/best_cnn_style_transfer_model.pth'):
        print("WARNING: Kaggle CNN model not found. This application requires pre-trained models from Kaggle.")
        print("Please ensure the kaggle_notebook_outputs directory contains best_cnn_style_transfer_model.pth")
    
    if os.path.exists('enhanced_best_style_transfer_model.pth'):
        try:
            os.remove('enhanced_best_style_transfer_model.pth')
            print("Removed legacy enhanced model file.")
        except:
            print("Could not remove legacy enhanced model file. You may want to delete it manually.")
            
    if os.path.exists('enhanced_best_vit_style_transfer_model.pth'):
        try:
            os.remove('enhanced_best_vit_style_transfer_model.pth')
            print("Removed legacy enhanced ViT model file.")
        except:
            print("Could not remove legacy enhanced ViT model file. You may want to delete it manually.")
    
    app.run(debug=True) 