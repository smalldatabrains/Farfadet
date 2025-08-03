import torch
from ViTSegmentor import VITSegmentor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = VITSegmentor(num_classes=33).to(device)
model.load_state_dict(torch.load('model\\vit_segmentation_epoch_360.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model.eval()

from PIL import Image
from torchvision import transforms

# Replace with your image path
image_path = 'test.jpg'

transform = transforms.Compose({
    transforms.Resize((224,224)),
    transforms.ToTensor()
})

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

input_tensor = input_tensor.to(device)

with torch.no_grad():
    output = model(input_tensor)
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(predicted_mask)
plt.title("Predicted Mask")
plt.axis('off')

plt.tight_layout()
plt.show()