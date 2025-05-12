# Neural Style Transfer Web Application

This project implements a Neural Style Transfer web application that allows users to upload content images and apply New Realism artistic style using pre-trained models.

## Features

- Upload content images
- Adjust style transfer intensity with a slider
- Choose between CNN and Vision Transformer (ViT) models
- Download generated stylized images

## Project Structure

```
Style-Transfer-CNN/
│
├── app.py                            # Flask application
├── models/                           # Model implementations
│   ├── __init__.py
│   ├── cnn_model.py                  # CNN-based style transfer model
│   └── vit_model.py                  # Vision Transformer style transfer model
├── utils/                            # Utility functions
│   ├── __init__.py
│   └── transforms.py                 # Image transformation utilities
├── static/                           # Static files
│   └── uploads/                      # Upload directory for images
├── templates/                        # HTML templates
│   └── index.html                    # Main application page
├── style_transfer_kaggle.py          # Kaggle implementation
├── kaggle_notebook_outputs/          # Pre-trained models (not included in repo)
└── requirements.txt                  # Project dependencies
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. **Download the pre-trained models:**
   - The pre-trained model files are too large for GitHub
   - Download them from [this link](https://drive.google.com/file/d/1lVS8UKYG0tomg-iNNXfNs0sc_WPvMhaT/view?usp=sharing) and place them in a folder named `kaggle_notebook_outputs`
   - Required model files:
     - best_cnn_style_transfer_model.pth
     - best_vit_style_transfer_model.pth

## Usage

1. Run the Flask application:

```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`
3. Upload a content image (the image you want to stylize)
4. Adjust the style intensity using the slider
5. Select the model type (CNN or ViT)
6. Click "Generate Stylized Image" and wait for the result
7. Download the stylized image if desired

## Model Architecture

### CNN Model
- Uses VGG19 for feature extraction
- Includes built-in style intensity control
- Content representation: conv4_2
- Style representation: multiple VGG layers
- Custom decoder for image generation

### Vision Transformer Model
- Integrates Vision Transformer for style feature extraction
- Uses the same decoder architecture as the CNN model for compatibility
- Combines the strengths of CNNs and Transformers for style transfer
- Leverages ViT's ability to capture global contextual information

## Comparing CNN and ViT Approaches

The CNN model follows the traditional approach to style transfer, using VGG19 convolutional layers to extract features for both content and style. It works well for capturing local textures and patterns.

The Vision Transformer model offers an alternative approach, using transformers for style representation. Transformers excel at capturing global relationships between different parts of an image, potentially leading to more coherent stylization with better global structure. 
