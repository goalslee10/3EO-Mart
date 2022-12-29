from django.shortcuts import render
from .models import Product
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import numpy as np


def index(request):
    product_list = Product.objects.order_by('id')
    product = {'product_list': product_list}

    return render(request, 'listpage/product_list.html', product)

INPUT_WIDTH = 640                       # ()
INPUT_HEIGHT = 640                      #
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3                   #   45
CONFIDENCE_THRESHOLD = 0.3            #   45  정확도  수치
# 폰트들
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
# 색깔지정   ( R  ,  G   , B )
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)
GREEN = ( 0 , 255 , 0)

class VideoCamera(object):   

    def draw_label(input_image, label, left, top):
        # 텍스트 사이즈 조정
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # 텍스트를 둘러싼 검은 상자.
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
        # 내부에 텍스트 사용.
        cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
        # OpenCV의 dnn 모듈이 처리할 수 있는 이미지 Blob에
        # 학습된 모델의 조건을 기준으로 프레임별로 너비, 높이, 0~255 사이 레벨 조정, BGR 평균 넣기
    def pre_process(input_image, net):
        # 4D BLOB화
        blob = cv2.dnn.blobFromImage(input_image, 1.0/255., (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], swapRB=True, crop=False)
        # DNN에 프레임 넣기
        net.setInput(blob)
        # # GPU가 있으면 아래 두 코드를 사용해주세요
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # 순방향 네트워크 실행
        output_layers = net.getUnconnectedOutLayersNames()
        # print(output_layers)
        # 전체 layer에서 실제 detection이 완료된 레이어를 출력
        outputs = net.forward(output_layers)
        print(outputs[0].shape)
        return outputs

    def post_process(input_image, outputs):
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # 이미지 resize
        x_factor = image_width / INPUT_WIDTH
        y_factor =  image_height / INPUT_HEIGHT
        # 하나의 행씩 값을 받음. 전체 85 중 앞의 4개는 바운딩박스의 좌표 관련 값
        # 그 다음은 objectness Score, 나머지 80개는 Class Scores(80개 클래스에 대한 각각의 확률값)
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Confidence가 우리가 정한 CONFIDENCE_THRESHOLD보다 높으면 검출하기로 결정
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                # class score가 가장 높은 좌표의 인덱스번호 찾기
                class_id = np.argmax(classes_scores)
                #  해당 score가 우리가 정한 score threshold보다 높으면 경계 출력
                if (classes_scores[class_id] > SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/3) * x_factor)     #  수정  ( /2 )
                    top = int((cy - h/3) * y_factor)      #  수정  ( /2 )
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        # NMS - non maximum suppression: 박스 여러개가 겹칠 경우 가장 높은 confidence를 가진 하나만을 검출
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for i in indices:
            box    = boxes[i]
            left   = box[0]
            top    = box[1]
            width  = box[2]
            height = box[3]
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            label = "{}:{:.2f}".format(VideoCamera.classes[class_ids[i]], confidences[i])
            VideoCamera.draw_label(input_image, label, left, top)
        return input_image

    # 웹캠 신호 받기
cap = cv2.VideoCapture(0)    # 0은 현재 노트북 내장 , 1 이상은 usb 장착된 폰카
# 학습시킨 라벨별 클래스명
classesFile = 'listpage/coconames.txt'  #  마트 목록 리스트 작성
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# 학습시켰던 ONNX 파일을 모델에 넣기
modelWeights = "listpage/best_yolov5m_plus_batch16.onnx"
# modelWeights = "best.onnx"
net = cv2.dnn.readNet(modelWeights)
while True:
    # 웹캠 프레임
    ret, frame = cap.read()
    h, w, c = frame.shape
    detections = VideoCamera.pre_process(frame, net)
    img = VideoCamera.post_process(frame.copy(), detections)
    # getPerfProfile() : 실행시간 계산에 관련된 함수. 추론에 소요된 전체 틱 시간과 각 레이어에서 소요된 틱 시간을 반환
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    print(label)
    cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
    cv2.imshow('Output', img)
    # q 버튼 누르면 종료
    if cv2.waitKey(1) == 0xFF & ord('q'):
        break
# 전체 자원 반환
cap.release()
    
    # def __init__(webcam):
    #     webcam.video = cv2.VideoCapture(0)
    #     classesFile = 'coconames.txt'  #  마트 목록 리스트 작성
    #     classes = None
    #     with open(classesFile, 'rt') as f:
    #         classes = f.read().rstrip('\n').split('\n')
    #     # 학습시켰던 ONNX 파일을 모델에 넣기
    #     modelWeights = "best_yolov5m_plus_batch16.onnx"
    #     # modelWeights = "best.onnx"
    #     net = cv2.dnn.readNet(modelWeights)

    #     threading.Thread(target=webcam.update, args=()).start()

#     def __del__(webcam):
#         webcam.video.release()

#     def get_frame(webcam):
#         image = webcam.frame
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def update(webcam):
#         while True:
#             (webcam.grabbed, webcam.frame) = webcam.video.read()
    
#     def gen(camera):
#         while True:
#             frame = camera.get_frame()
#         # frame단위로 이미지를 계속 반환한다. (yield)
#             yield(b'--frame\r\n'
#                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# # detectme를 띄우는 코드(여기서 웹캠을 킨다.)
# @gzip.gzip_page
# def listpage(request):
#     try:
#         cam = VideoCamera() #웹캠 호출
#         # frame단위로 이미지를 계속 송출한다
#         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
#     except:  # This is bad! replace it with proper handling
#         print("에러입니다...")
#         pass

# class VideoCamera(object):
#     def __init__(webcam):
#         webcam.video = cv2.VideoCapture(0)
#         (webcam.grabbed, webcam.frame) = webcam.video.read()
#         threading.Thread(target=webcam.update, args=()).start()

#     def __del__(webcam):
#         webcam.video.release()

#     def get_frame(webcam):
#         image = webcam.frame
#         _, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()

#     def update(webcam):
#         while True:
#             (webcam.grabbed, webcam.frame) = webcam.video.read()


# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         # frame단위로 이미지를 계속 반환한다. (yield)
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# # detectme를 띄우는 코드(여기서 웹캠을 킨다.)
# @gzip.gzip_page
# def listpage(request):
#     try:
#         cam = VideoCamera() #웹캠 호출
#         # frame단위로 이미지를 계속 송출한다
#         return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
#     except:  # This is bad! replace it with proper handling
#         print("에러입니다...")
#         pass
