import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model_training import ConvNet

def load_model():
    model = ConvNet(num_classes=3)
    model.load_state_dict(torch.load("ConvNet1.pth"))
    model.eval()
    return model

def load_image(image_path):
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(original_image).unsqueeze(0)  # shape: [1, 3, 256, 256]
    return image_tensor, original_image, original_size

def show_image_and_mask(original_image, predicted_mask):
    predicted_mask = predicted_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_image)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(predicted_mask, cmap="jet", interpolation="nearest")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image_tensor, original_image, original_size = load_image("data\\corrobot.v2i.coco-segmentation\\valid\\S7_jpg.rf.7895a4a6f52b6beb3a1ec20e09f50cb2.jpg")
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)  # output shape: [1, C, H, W]
        predicted_mask = torch.argmax(output, dim=1, keepdim=True).float()  # shape: [1, 1, H, W]

        # Upsample mask to original image size
        predicted_mask = F.interpolate(predicted_mask, size=(original_size[1], original_size[0]), mode='nearest')
        predicted_mask = predicted_mask.squeeze().long()  # [H, W]

    show_image_and_mask(original_image, predicted_mask)
