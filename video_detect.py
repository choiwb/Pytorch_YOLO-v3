import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2
from models import *
from utils.utils import *
from utils.datasets import *

#############################
from sort import *
#############################

config_path = 'config/yolov3_custom.cfg'
weights_path = 'checkpoints/yolov3_ckpt_9.pth'
class_path = 'data/custom/classes.names'

img_size=480
conf_thres=0.5
nms_thres=0.4

# Tensor = torch.FloatTensor
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

classes = load_classes(class_path)  # Extracts class labels from file
print(classes)

model = Darknet(config_path, img_size=img_size)

if weights_path.endswith(".pth"):
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))
else:
    # Load darknet weights
    model.load_darknet_weights(weights_path)
    
model = model.cuda() if torch.cuda.is_available() else model
    
mot_tracker = Sort()

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
    return detections[0]

videopath = 'data/circuit_2.mp4'
vid = cv2.VideoCapture(videopath)

# Red color 제외!
colors=[(255,0,0),(0,255,0),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

cv2.resizeWindow('Stream', (480, 480))

ret, frame = vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print("Video size: (%d, %d)" %(vw, vh))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

while(True):
    ret, frame = vid.read()
        
    if not ret:
        break
    
    # frame 관련 2가지 중복 코드 (아래 코드와 중복 필요) BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    # print(detections)
    
    # frame 관련 2가지 중복 코드 (아래 코드와 중복 필요) RGB -> BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    img = np.array(pilimg)
    # print(img)
    
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    
    if detections is not None:

        tracked_objects = mot_tracker.update(detections.cpu())
        # box = detections[0:4] * Tensor([vw, vh, vw, vh])
        box = detections[0:4]
        print(box)

        unique_labels = detections[:, -1].cuda().unique()
        
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)

        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

            # bounding box 별로 color 다르게 잡힘!
            color = colors[int(cls_pred) % len(colors)]
            cls = classes[int(cls_pred)]

            # all bounding box per 1 frame
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)

            # object label box
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 30, y1), color, -1)

            # object label name
            # cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #            (255, 255, 255), 3)
            cv2.putText(frame, cls, (x1 + 15, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 3)
    
    cv2.imshow('Stream',frame)
    outvideo.write(frame)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

cv2.destroyAllWindows()
outvideo.release()