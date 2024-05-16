from PIL import Image
import matplotlib.pyplot as plt

# Example image paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Calculate number of rows and columns for subplot grid
num_images = len(image_paths)
num_cols = 3  # Number of columns for displaying images
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

# Load and display images
for i, image_path in enumerate(image_paths):
    # Load image
    image = Image.open(image_path)
    
    # Calculate subplot index
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    # If there's only one row, axes will not be a 2D array
    if num_rows == 1:
        ax = axes[col_idx]
    else:
        ax = axes[row_idx, col_idx]
    
    # Display the image on the subplot
    ax.imshow(image)
    ax.axis('off')

# Hide any remaining empty subplots
for i in range(num_images, num_rows * num_cols):
    if num_rows == 1:
        fig.delaxes(axes[i])
    else:
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axes[row_idx, col_idx])

# Adjust layout and display the figure
plt.tight_layout()
plt.show()
