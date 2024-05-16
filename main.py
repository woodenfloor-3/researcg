import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ViT model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Load pre-trained ResNet model
resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
resnet_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")

# Move models to the appropriate device
model.to(device)
resnet_model.to(device)

# Preprocess an image for ViT model
def preprocess_image_vit(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

# Preprocess an image for ResNet model
def preprocess_image_resnet(image_path):
    image = Image.open(image_path)
    inputs = resnet_processor(images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}

# Example usage
image_path = "image.jpg"
true_label = 0  # Example true label for the image

# Perform inference for ViT
inputs_vit = preprocess_image_vit(image_path)
with torch.no_grad():
    outputs_vit = model(**inputs_vit)
    predicted_label_vit = torch.argmax(outputs_vit.logits, dim=1).item()

# Perform inference for ResNet
inputs_resnet = preprocess_image_resnet(image_path)
with torch.no_grad():
    outputs_resnet = resnet_model(**inputs_resnet)
    predicted_label_resnet = torch.argmax(outputs_resnet.logits, dim=1).item()

# Convert predicted labels to string labels if needed
label_map = ["panda", "tiger", "leaves"]  # Example label mapping
predicted_label_vit_str = label_map[predicted_label_vit]
predicted_label_resnet_str = label_map[predicted_label_resnet]

# Print predicted labels
print("Predicted label (ViT):", predicted_label_vit_str)
print("Predicted label (ResNet):", predicted_label_resnet_str)
