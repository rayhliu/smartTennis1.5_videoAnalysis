import os 
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import base64

from court_detector import CourtDetector
from tennis_detector.predict_tracknet_v2 import TennisDetector
from players_detector.tennis_players_detector import TennisPlayersDetector
from video_reader import VideoReader


def loading_bar(current_idx,total_size):
    total_size-=1
    print('\r' + '[Progress:] [%s%s] %.2f%%;' % (
        '█' * int(current_idx*20/total_size), 
        ' ' * (20-int(current_idx*20/total_size)),
        float(current_idx/total_size*100)), end='')
    return int(current_idx/total_size*100)

def convert_img2base64(cv_img):
    retval, buffer = cv2.imencode('.png', cv_img)
    img_base64 = base64.b64encode(buffer).decode('UTF-8')
    return img_base64

class TennisSinglesApp:
    def __init__(self,video_path, save_info=False, show_info=False, verbose=False):
        video_dir, video_name = video_path.split('/')[-2:]
        self.session_name = video_dir+":"+video_name
        print ('session_name:',self.session_name)
        self.response_result = {
            'state':None,
            'session_name':self.session_name,
            'court_score':0,
            'bounced_info':{},
            'img_base64':None
            }

        if os.path.isfile('assert/court_2d_base64.txt'):
            with open('assert/court_2d_base64.txt','r') as f:
                self.response_result['img_base64'] = f.read()
        
        try:
            self.video_reader = VideoReader(video_path)
        except:
            if self.response_result['state'] is None:
                self.response_result['state'] = 'failed to init file reader'
                print ('[ERROR] Faild to init file reader.')

        self.img = None
        self.img_list = [] # for demo
        self.show_info = show_info
        self.save_info = save_info
        self.do_palyer_detect = False  
        self.court_detect_scale = 2
        self.running_state = None
        self.court_thresh = 0.2

        self.save_dir = None
        if self.save_info:
            self.save_dir = os.path.dirname(video_path)

        try:
            self.init_court_detector()
        except:
            if self.response_result['state'] is None:
                self.response_result['state'] = 'faild to init court detector.'
                print ('[ERROR] Faild to init court detector.')

        try:
            self.init_detector_model()
        except:
            if self.response_result['state'] is None:
                self.response_result['state'] = 'failed to init_model.'
                print ('[ERROR] Faild to init_model.')


        self.last_is_game_running = False

        # In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames 
        self.show_ball_list = [None]*8

        self.final_bounced_info = None
        self.verbose = verbose
    
    def init_detector_model(self):
        """ init trt context and runtime """
        self.core_context = cuda.Device(0).make_context()
        core_logger = trt.Logger(trt.Logger.WARNING)
        core_runtime = trt.Runtime(core_logger)
        share_trt_core = [self.core_context,core_logger,core_runtime]

        print ("init tennis detrctor model...")
        self.TD = TennisDetector(shareTRTCore=share_trt_core)

        if self.do_palyer_detect:
            print ("init tennis player detrctor model...")
            self.TPD = TennisPlayersDetector(
                'players_detector/weights/yolov7-nms.trt',
                courtInfo=self.court_info,
                shareTRTCore=share_trt_core)

    def init_court_detector(self):
        """ init court detector """
        court_detector = CourtDetector()
        print ('Detect tennis court...')
        _,_,init_frame = self.video_reader.get_frame()
        resize_init_frame = init_frame[::self.court_detect_scale,::self.court_detect_scale,:]
        self.court_info = court_detector.get_court_lines(0,resize_init_frame)

        pts = []
        lines = self.court_info[0]
        for i in range(0, len(lines), 4):
            if i in [0,4]:
                pts += [[lines[i],lines[i+1]],[lines[i+2],lines[i+3]]]
        pts = np.array(pts)
        court_2d_area = (max(pts[:,0])-min(pts[:,0])) * (max(pts[:,1])-min(pts[:,1]))
        court_2d_score = round(court_2d_area/(resize_init_frame.shape[0]*resize_init_frame.shape[1]),4)
        print ('Court detect score:',court_2d_score)
        self.response_result['court_score'] = court_2d_score
        if court_2d_score <=self.court_thresh:
            self.court_info = None
        
        if self.show_info:
            for i in range(0, len(lines), 4):
                lc = (0,0,255)
                if i == 8: 
                    lc = (255,0,0)
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                resize_init_frame = cv2.line(resize_init_frame, (int(x1),int(y1)),(int(x2),int(y2)), lc, 3)

            cv2.namedWindow('court_line_detect',cv2.WINDOW_NORMAL)
            cv2.imshow('court_line_detect',resize_init_frame)
            cv2.waitKey(0)

    def _draw_tracked_circles(self,img,tracked_circles):  
        # draw court line
        if self.court_info is not None:
            lines = self.court_info[0]
            for i in range(0, len(lines), 4):
                lc = (0,0,255)
                if i == 8: lc = (255,0,0)
                x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
                cv2.line(img, (int(x1),int(y1)),(int(x2),int(y2)), lc, 5)

        # process show balls 
        if len(tracked_circles) > 0:
            tracked_circle, tracked_times, _ = max(tracked_circles,key= lambda x:x[1])
            if tracked_times>1:
                self.show_ball_list.append([int(tracked_circle[0]),int(tracked_circle[1])])
                self.show_ball_list.pop(0)
            else:
                self.show_ball_list.append(None)
                self.show_ball_list.pop(0)
        else:
            self.show_ball_list.append(None)
            self.show_ball_list.pop(0)
        
        # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
        for ball_idx, ball in enumerate(self.show_ball_list):
            if ball is not None:
                if ball_idx == 7:
                    cv2.circle(img,(ball[0],ball[1]),7,(0,0,255),2)
                else:
                    cv2.circle(img,(ball[0],ball[1]),7,(0,255,255),2)
        return img
    
    def _draw_2d_court_bounced_info(self,bounced_info):
        scale_rate = 3
        court_2d_img = self.court_info[1].court_2d
        court_2d_img = court_2d_img[::scale_rate,::scale_rate,:].copy()
        for key, value in bounced_info.items():
            if value['player'] == 'top':
                player_color = (0,0,255) 
            elif value['player'] == 'bottom':
                player_color = (255,255,255)
            else:
                player_color = (100,100,100)
            
            if value['state'] == 'in':
                ball_color = (0,200,0)
            elif value['state'] == 'out':
                ball_color = (200,200,200)
            else:
                ball_color = (255,255,255)

            b_x, b_y = value['bounced_2d_xy']
            bounced_xy = (b_x//scale_rate,b_y//scale_rate)
            cv2.circle(court_2d_img,bounced_xy,15,ball_color,-1)
            cv2.circle(court_2d_img,bounced_xy,16,(200,200,200),2)
            cv2.putText(court_2d_img, 
                str(key), 
                (bounced_xy[0]-8,bounced_xy[1]+10), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 
                player_color, 
                2, 
                cv2.LINE_AA)

        return court_2d_img

    def _draw_player_bboxes(self,img,top_player_bbox,bottom_player_bbox):
        if top_player_bbox is not None:
            cv2.rectangle(img,tuple(top_player_bbox[:2]),tuple(top_player_bbox[2:]),[255,0,0],5)
        if bottom_player_bbox is not None:
            cv2.rectangle(img,tuple(bottom_player_bbox[:2]),tuple(bottom_player_bbox[2:]),[0,0,255],5)
        return img

    def excute(self):
        try:
            if self.response_result['state'] is not None: 
                raise ValueError ('bad init state')

            imgs_list = []
            timestamp_list = []
            while True:
                frame_count, timestamp, self.img = self.video_reader.get_frame()
                imgs_list.append(self.img)
                timestamp_list.append(timestamp)

                if self.img is None: break

                is_game_running = True

                """ player detect """
                if self.do_palyer_detect:
                    top_player_bbox = None
                    bottom_player_bbox = None
                    is_game_running, top_player_info, bottom_player_info = self.TPD.detect(self.img)
                    top_player_bbox, top_player_2d_court_xy = top_player_info[:2]
                    bottom_player_bbox, bottom_player_2d_court_xy = bottom_player_info[:2]
                
                """ tennis_tracking """
                if is_game_running:
                    tracked_circles = self.TD.detect(self.img)
                
                if self.last_is_game_running == False and is_game_running == True:
                    # TODO start record  tennis
                    pass
                
                if self.last_is_game_running == True and is_game_running == False:
                    # TODO init init tennis record
                    pass

                self.last_is_game_running = is_game_running
                
                if self.show_info:
                    demo_img = self.img.copy()
                    demo_img = self._draw_tracked_circles(demo_img,tracked_circles)
                    if self.do_palyer_detect:
                        demo_img = self._draw_player_bboxes(demo_img,top_player_bbox,bottom_player_bbox)

                    # cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
                    # cv2.imshow('demo',demo_img)
                    # cv2.waitKey(0)

                loading_rate = loading_bar(frame_count,self.video_reader.video_count)
                self.running_state = "session:{} \nAnalysing...{}%".format(self.session_name,str(loading_rate))
                
            print ('\n')
            
            if self.save_info:
                self.running_state += "\nCreating demo video..."

            """ process ball bounced """
            final_bounced_info = self.TD.get_point_bounced_info(
                court_info=self.court_info,
                imgs=imgs_list,
                timestamps=timestamp_list, 
                saveVideoPath=self.save_dir,
                frame_scale=self.court_detect_scale,
                verbose=self.verbose)

            # print (final_bounced_info)
            self.response_result['bounced_info'] = final_bounced_info
            self.response_result['state'] = 'ok'
            
            if self.court_info is not None:
                court_2d_img = self._draw_2d_court_bounced_info(final_bounced_info)
                base64_img = convert_img2base64(court_2d_img)
                self.response_result['img_base64'] = base64_img
        
                if self.show_info:
                    cv2.namedWindow('court_2d',cv2.WINDOW_NORMAL)
                    cv2.imshow('court_2d',court_2d_img)
                    cv2.waitKey(0)
            
            self.core_context.pop()

            self.running_state += "\nFinished"
            print (self.running_state)
            
            return self.response_result

        except:
            if self.response_result['state'] is None:
                self.response_result['state'] = 'faild to get analysis info'

            return self.response_result

if __name__ == "__main__":
    app = TennisSinglesApp('./session/video.mp4')
    app.excute()