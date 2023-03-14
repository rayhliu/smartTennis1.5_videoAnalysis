import cv2
import numpy as np

class VideoReader:
    def __init__(self,video_path,verbose=False):
        self.verbose = verbose

        self.video = cv2.VideoCapture(video_path)
        video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print ('video_fps:{} | video_count:{}'.format(video_fps,self.video_count))
        if self.video_count == 0:
            raise ValueError ('invaild video path: %s'%video_path)
        self.same_frame_bug = False
        self.same_frmae_bug_start_count = 0
        self.jump_frame_size = int(max(video_fps//30, 1))
        print ('jump_frame_size:',self.jump_frame_size)

        self.last_img = None

        self.frame_count = 0

    def check_same_frame_bug(self,img):
        if self.last_img is not None:
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray_last_img = cv2.cvtColor(self.last_img,cv2.COLOR_BGR2GRAY)
            diffMap = cv2.absdiff(gray_img,gray_last_img)
            diffMask = cv2.threshold(diffMap,10,1,cv2.THRESH_BINARY)[1]
            same_rate = np.sum(diffMask)/diffMask.size
            if same_rate < 3e-4:
                self.same_frame_bug = True

        self.last_img = img.copy()

    def get_frame(self):
        img = None
        frame_count =  int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_count < self.video_count:
            for _ in range(self.jump_frame_size):
                ret, img = self.video.read()
                if self.verbose:
                    print ('current_frame_count:',frame_count)

                if frame_count <= 7:
                    if self.same_frame_bug == False and self.same_frmae_bug_start_count == 0:
                        self.check_same_frame_bug(img)

                        if self.same_frame_bug:
                            self.same_frmae_bug_start_count = frame_count
                
                if self.same_frame_bug:
                    if (frame_count-self.same_frmae_bug_start_count)%6 == 0:
                        if self.verbose:
                            print (frame_count,'is ignored.')
                        ret, img = self.video.read()

        frame_count =  int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = int(self.video.get(cv2.CAP_PROP_POS_MSEC))
        return frame_count, timestamp, img