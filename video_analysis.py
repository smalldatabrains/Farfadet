import cv2 as cv
import torch.nn as nn
import torch
from model_training import ConvNet #from training file classifier.py
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class ModelLoader(nn.Module):
    """
    A class to load a segmentation model and apply inference on a frame
    """
    def __init__(self,path="ConvNet2.pth"):
        super(ModelLoader,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model= ConvNet(num_classes=2).to(self.device) # simple classifier
        self.model.load_state_dict(torch.load(path, map_location=self.device)) #load previously trained model weights and parameters
        self.model.eval() # eval mode to speed up the process
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def forward(self, frame):
        # Preprocess
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)  # [1, 3, 256, 256]
        with torch.no_grad():
            output = self.model(input_tensor)  # shape: [1, 3, 256, 256]
            predicted_mask = torch.argmax(output, dim=1, keepdim=True).float()  # [256, 256]
            predicted_mask = F.interpolate(predicted_mask,size=(320, 320), mode='nearest')  # back to patch size
            predicted_mask = predicted_mask.squeeze().long().cpu()  # [H, W]
        return predicted_mask

class VideoLoader ():
    """
    A class to load, display, and process videos for inference.

    Attributes:
        path (str): Path to the input video.
        output_folder (str): Directory to save output files.
        mode (str): Mode of operation (default is 'inference').
        HEIGHT (int): Height for resizing frames.
        WIDTH (int): Width for resizing frames.
    """
    def __init__(self, path):
        self.path=path
        self.output_folder='.\\data\\videos\\outputs'
        self.mode = 'inference'
        self.HEIGHT=1540
        self.WIDTH=2560
        self.model=ModelLoader()

    def read_video(self):
        capture = cv.VideoCapture(self.path)
        if not capture.isOpened():
            print("Error opening video")
            return None
        
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            resized_frame = cv.resize(frame,(self.WIDTH, self.HEIGHT))
            patches=self.cut_frame(resized_frame)


            for id,patch in enumerate(patches):
                pil_patch=Image.fromarray(cv.cvtColor(patch, cv.COLOR_BGR2RGB))
                predicted_mask = self.model(pil_patch)
                mask_np=predicted_mask.cpu().numpy().astype('uint8')*80
                # print(predicted_mask.shape)
                # Display side by side
                color_mask = cv.applyColorMap(mask_np, cv.COLORMAP_JET)
                if id == 9 or id == 16:
                    cv.imshow(f'Patch {id}', patch)
                    cv.imshow(f'Mask {id}',color_mask)

            keyboard=cv.waitKey(30)
            if keyboard == ord('q') or keyboard == 27:
                break
        capture.release()
        cv.destroyAllWindows()

    def cut_frame(self,frame):
        """
        Return a list of small images patches 100x100 from an HD frame to be sent next to the model
        """
        patches=[]
        small_frame_height=320
        small_frame_width=320
        
        for y in range(0, frame.shape[0],small_frame_height):
            for x in range(0, frame.shape[1], small_frame_width):
                patch = frame[y:y+small_frame_height,x:x+small_frame_width]
                if patch.shape[0] == small_frame_height and patch.shape[1] == small_frame_width:
                    patches.append(patch)

        return patches

    def inference(self, frame):
        """
        Return segmentation mask and some metrics
        """
        return self.model(frame)
        

    def save_measurements(self):
        """
        Save some key metrics after analysis
        """
        pass



if __name__ == '__main__':
    video=VideoLoader('.\\data\\videos\\4721836-hd_1920_1080_30fps.mp4')
    video.read_video()
    



