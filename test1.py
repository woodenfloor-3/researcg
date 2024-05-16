from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load ViT model
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Load ResNet-50 model
resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
resnet_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

# Open the image
image = Image.open("x.jpg")

# Preprocess the image for ViT
vit_input = vit_processor(images=image, return_tensors="pt")
vit_output = vit_model(**vit_input)

# Preprocess the image for ResNet-50
resnet_input = resnet_processor(images=image, return_tensors="pt")
resnet_output = resnet_model(**resnet_input)

# Get the predictions and log probabilities
vit_preds = vit_output.logits.softmax(dim=1).detach().cpu().numpy()[0]
resnet_preds = resnet_output.logits.softmax(dim=1).detach().cpu().numpy()[0]

# Print the top 5 predictions for each model
print("ViT Top Predictions:")
print(dict(sorted(zip(vit_model.config.id2label.values(), vit_preds), key=lambda x: x[1], reverse=True)[:5]))

print("\nResNet-50 Top Predictions:")
print(dict(sorted(zip(resnet_model.config.id2label.values(), resnet_preds), key=lambda x: x[1], reverse=True)[:5]))