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
    def __init__(self,path="ConvNet1.pth"):
        super(ModelLoader,self).__init__()

        self.model= ConvNet(num_classes=3) # simple classifier
        self.model.load_state_dict(torch.load(path)) #load previously trained model weights and parameters
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def forward(self, frame):
        # Preprocess
        input_tensor = self.transform(frame).unsqueeze(0)  # [1, 3, 256, 256]
        with torch.no_grad():
            output = self.model(input_tensor)[0]  # shape: [1, 3, 256, 256]
            predicted_mask = torch.argmax(output, dim=1).squeeze(0)  # [256, 256]
            predicted_mask = F.interpolate(predicted_mask.unsqueeze(0).unsqueeze(0).float(),
                                           size=(100, 100), mode='nearest')  # back to patch size
            predicted_mask = predicted_mask.squeeze().byte().cpu().numpy()  # [100, 100]
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
        self.HEIGHT=400
        self.WIDTH=600
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
                predicted_mask = self.model(patch)
                # Display side by side
                color_mask = cv.applyColorMap(predicted_mask, cv.COLORMAP_JET)
                stacked = cv.hconcat([patch, color_mask])
                cv.imshow(f'Patch {id}', stacked)

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
        small_frame_height=100
        small_frame_width=100
        
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
    video=VideoLoader('.\\data\\videos\\13175373-uhd_3840_2160_30fps.mp4')
    video.read_video()
    



