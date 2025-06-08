import cv2 as cv
import torch.nn as nn
from model_training import ConvNet, SimpleNet #from training file classifier.py

class ModelLoader(nn.Module):
    """
    A class to load a segmentation model and apply inference on a frame
    """
    def __init__(self,path):
        super(ModelLoader,self).__init__()

        self.classifier= SimpleNet() # simple classifier
        self.load_state_dict(state_dict='SimpleNet.pth') #load previously trained model weights and parameters

        self.model=ConvNet() # segmentation model
        self.load_state_dict(state_dict='UNET.pth') #load previsouly trained model weights and parameters
        self.eval() #evalutation mode of the model for inference

    def forward(self, frame):
        return self.classifer(frame), self.model(frame) #this return a mask

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
            cv.imshow('Resized frame', resized_frame)

            for id,patch in enumerate(patches):
                cv.imshow('Patch'+str(id), patch)

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
    



