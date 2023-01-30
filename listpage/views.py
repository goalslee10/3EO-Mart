from django.shortcuts import render
from django.http import HttpResponse
from .models import Product, Basket
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.core.paginator import Paginator
import cv2
from pathlib import Path
import threading
import numpy as np

def index(request):
    page = request.GET.get('page', '1')
    product_list = Product.objects.all()
    paginator = Paginator(product_list, 7)
    page_obj = paginator.get_page(page)
    product = {'product_list': page_obj}

    print(product_list)
    
    return render(request, 'listpage/product_list.html', product)
# def index(request):
#     # product = {'product_list': output_basket}
#     product_list = Product.objects.all()
#     product = {'product_list': product_list}

#     gen()
    
#     return render(request, 'listpage/product_list.html', product)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.3

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)
Green = ( 0 , 255 , 0)

BASE_DIR = Path(__file__).resolve().parent
classesFile = str(BASE_DIR)+'/coco.names'
classes = None

basket = []
output_basket=[]

modelWeights = str(BASE_DIR)+"/best_yolov5m_batch32_mix12.onnx"
net = cv2.dnn.readNet(modelWeights)

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
        
    def get_frame(self):

        input_image = self.frame

        blob = cv2.dnn.blobFromImage(input_image, 1.0/255., (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], swapRB=True, crop=False)
        net.setInput(blob)

        # # GPU가 있으면 아래 두 코드를 사용해주세요
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        output_layers = net.getUnconnectedOutLayersNames()
        # print(output_layers)
        outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        x_factor = image_width / INPUT_WIDTH
        y_factor =  image_height / INPUT_HEIGHT

        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)

                if (classes_scores[class_id] > SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/3) * x_factor)
                    top = int((cy - h/3) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for i in indices:
            box    = boxes[i]
            left   = box[0]
            top    = box[1]
            width  = box[2]
            height = box[3]
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            basket.append(classes[class_ids[i]])
            draw_label(input_image, label, left, top)

        _, jpeg = cv2.imencode('.jpg', input_image)
        finish = (jpeg.tobytes(), basket)
        return finish

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame[0] + b'\r\n\r\n')
        # print(frame[1])

@gzip.gzip_page
def listpage(request):
    try:
        cam = VideoCamera() 
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        print("에러입니다...")
        pass