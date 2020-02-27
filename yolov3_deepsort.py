import os
import cv2
import time
import argparse
import torch
import numpy as np
import csv

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = False
        #use_cuda = args.use_cuda and torch.cuda.is_available()
        #if not use_cuda:
        #    raise UserWarning("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names


    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def posprocessing(self, writer, idx_frame, im, outputs, start):
        face_cascade = cv2.CascadeClassifier('demo/haarcascade_frontalface_default.xml')

        for row in outputs:
            x = row[0]
            y = row[1]
            w = row[2]
            h = row[3]
            identity = row[4]
            video_name = self.args.VIDEO_PATH.split(".")[0]

            crop_image = im[int(y):int(h), int(x):int(w)]

            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face = None
            for (i, j, k, l) in faces:
                face = crop_image[int(j):int(j + l), int(i):int(i + k)]

                try:
                    path = os.getcwd()
                    print(path + "/results/{}/{}".format(video_name, identity))
                    os.mkdir(path + "/results/{}/{}".format(video_name, identity))
                except:
                    print("")
                    
                if(face.any()):
                    cv2.imwrite("results/{}/{}/{}.jpg".format(video_name, identity, idx_frame), face)

            print(start%60)
            writer.writerow({'x': x, 'y': y, 'w': w, 'h': h, 'frame': idx_frame, 'code': identity})
        

    def run(self):
        idx_frame = 0
        
        path = os.getcwd()
        os.mkdir(path + "/results/{}".format(self.args.VIDEO_PATH.split(".")[0]))

        with open(path + '/results/{}/annotations.csv'.format(self.args.VIDEO_PATH.split(".")[0]), 'w', newline='') as csvfile:
            fieldnames = ['x', 'y', 'w', 'h', 'frame', 'code']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while self.vdo.grab(): 
                idx_frame += 1
                if idx_frame % self.args.frame_interval:
                    continue

                start = time.time()
                _, ori_im = self.vdo.retrieve()
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

                # do detection
                bbox_xywh, cls_conf, cls_ids = self.detector(im)
                if bbox_xywh is not None:
                    # select person class
                    mask = cls_ids==0

                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                    cls_conf = cls_conf[mask]

                    # do tracking
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                    fps = self.vdo.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
                    frame_count = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count/fps

                    self.posprocessing(writer, idx_frame, ori_im, outputs, duration)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]

                        ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                end = time.time()

                #print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)
                    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
