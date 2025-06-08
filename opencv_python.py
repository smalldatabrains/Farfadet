import cv2 as cv
import torch.nn as nn
from classifier import ConvNet #from training file classifier.py

class LoadModel(nn.Module):
    """
    A class to load a model and apply inference on a frame
    """
    def __init__(self,path):
        super(LoadModel,self).__init__()
        self.model=ConvNet()
        self.load_state_dict(state_dict='ConvNet.pth') #load weights and parameters
        self.eval() #evalutation mode of the model for inference
    def forward(self, frame):
        return self.model(frame) #this return a mask

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
        while True:
            ret, frame = capture.read()
            cv.imshow('Input',frame)

            keyboard=cv.waitKey(30)
            if keyboard =='q' or keyboard == 27:
                break
        capture.release()
        cv.destroyAllWindows()

    def inference(self, frame):
        pass

    def save_output(self):
        """
        Save output video to local disk
        """
        pass



if __name__ == '__main__':
    video=VideoLoader('.\\data\\videos\\13175373-uhd_3840_2160_30fps.mp4')
    video.read_video()
    



