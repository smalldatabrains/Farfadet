import torch
from ViTSegmentor import VITSegmentor
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = VITSegmentor(num_classes=33).to(device)
model.load_state_dict(torch.load('model\\vit_segmentation_epoch_9500.pth', map_location=device))
model.eval()

# Load and preprocess image
image_path = 'test.jpg'
image = Image.open(image_path).convert("RGB")
original_size = image.size  # Save original (W, H)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)

    # Resize model output to 224x224 (if needed)
    output = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)

    # Get predicted mask
    predicted_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

    # Resize predicted mask to original image size
    predicted_mask_resized = Image.fromarray(predicted_mask.astype(np.uint8)).resize(original_size, resample=Image.NEAREST)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask_resized)  # Use a colormap suitable for many classes
plt.title("Predicted Segmentation Mask")
plt.axis('off')

plt.tight_layout()
plt.show()
