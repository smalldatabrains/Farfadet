import cv2 as cv
import torch.nn as nn

class BuildModel(nn.Module):
    """
    A class to train and detect buildings on a video
    """
    def __init__(self,path):
        super(BuildModel,self).__init__()
        self.dataset_path=path
        self.model = nn.Sequential(
            nn.Conv2d(),
            nn.MaxPool2d(),
            nn.Conv2d(),
            nn.MaxPool2d(),
            nn.Linear(),
            nn.ReLU()
        )

    def forward(self, frame):
        return self.model(frame)    
    
    def train_model(self):
        self.train()




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

    def save_output(self):
        """
        Save output video to local disk
        """
        pass



if __name__ == '__main__':
    video=VideoLoader('.\\data\\videos\\13175373-uhd_3840_2160_30fps.mp4')
    video.read_video()
    



