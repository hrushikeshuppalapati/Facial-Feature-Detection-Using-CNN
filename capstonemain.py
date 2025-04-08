import mediapipe as mp 
import time
import math
import numpy as np 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import cv2 
close_eye_count =0
Blink_counts =0
Close_frames = 1
FONTS = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0,250, 0) 
face_outline=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
NOSE = [8,240,460]
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5)
def detect_eye_mouth_status(face_img, raw_img):
    global close_eye_count, Blink_counts
    rgb_frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) 
    results  = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        img_h, img_w= face_img.shape[:2]
        mesh_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in results.multi_face_landmarks[0].landmark]
        reRatio, leRatio, mRatio = Open_Close_Ratios(face_img, mesh_coords, RIGHT_EYE, LEFT_EYE) 
        ratio = round((reRatio+ leRatio)/2, 2) 
        print("MOUTH RATIO:", mRatio)
        print("Eye RATIO:", ratio)
        eye_threshold = 3.5      
        if mRatio > 1.8: 
            cv2.putText(raw_img, 'Mouth Closed', (10, 30), FONTS, 1, TEXT_COLOR, 1)
        else:
            cv2.putText(raw_img, 'Mouth Open', (10, 30), FONTS, 1, TEXT_COLOR, 1)
        if ratio > eye_threshold:
            cv2.putText(raw_img, 'Eyes Closed', (10, 70), FONTS, 1, TEXT_COLOR, 1)
        else:
            cv2.putText(raw_img, 'Eyes Open', (10, 70), FONTS, 1, TEXT_COLOR, 1)
        if ratio > eye_threshold:
            close_eye_count +=1
        else:
            if close_eye_count>Close_frames:
                Blink_counts +=1
                close_eye_count =0
        cv2.putText(raw_img, f'Total Blinks: {Blink_counts}', (10, 110), FONTS, 1, TEXT_COLOR, 1) 
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in LIPS ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in face_outline ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in LEFT_EYEBROW ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in RIGHT_EYEBROW ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in NOSE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
    return face_img, raw_img
def Open_Close_Ratios(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[246]  
    rh_left = landmarks[133] 
    rv_top = landmarks[160]
    rv_bottom = landmarks[145]
    lh_right = landmarks[362]
    lh_left = landmarks[387]
    lv_top = landmarks[386]
    lv_bottom = landmarks[374]
    rhDistance = math.dist(rh_right, rh_left)
    rvDistance = math.dist(rv_top, rv_bottom)
    lvDistance = math.dist(lv_top, lv_bottom)
    lhDistance = math.dist(lh_right, lh_left)
    try:
        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance
    except:
        reRatio = 0
        leRatio = 0
    mouth_right = landmarks[409] 
    mouth_left = landmarks[185]
    mouth_top = landmarks[0]
    mouth_bottom = landmarks[17]
    mhDistance = math.dist(mouth_right, mouth_left)
    mvDistance = math.dist(mouth_top, mouth_bottom)
    try:
        mRatio = mhDistance/mvDistance
    except:
        mRatio = 10
    return reRatio,  leRatio, mRatio
classes = ['Male','Female']
model = load_model(r"E:\CAPSTONE\MULTI FEATURE EVALUATION\models\gender_detection.model")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_json = r"E:\CAPSTONE\MULTI FEATURE EVALUATION\models\face_emotion_model.json"
emotion_weight = r"E:\CAPSTONE\MULTI FEATURE EVALUATION\models\face_emotion_model.h5"
with open(emotion_json, "r") as json_file:
    loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(emotion_weight)
print("Emotion Model loaded")
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ageProto = r"E:\CAPSTONE\MULTI FEATURE EVALUATION\models\age_deploy.prototxt"
ageModel = r"E:\CAPSTONE\MULTI FEATURE EVALUATION\models\age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageNet = cv2.dnn.readNet(ageModel,ageProto) 

cam = cv2.VideoCapture(0)

while True:
    _, raw_img = cam.read() 
    if _:
        start_time = time.time() 
        print("Raw Image:", raw_img.shape)
        x,y,w,h = 0,0,0,0
        img_w = raw_img.shape[1]
        img_h = raw_img.shape[0]

        face_detection_results = face_detection.process(raw_img[:,:,::-1])

        if face_detection_results.detections:
            for face in face_detection_results.detections:

                print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
                if face.score[0] < 0.8: 
                    continue

                face_data = face.location_data

                x,y,w,h = int(img_w*face_data.relative_bounding_box.xmin), \
                            int(img_h*face_data.relative_bounding_box.ymin), \
                            int(img_w*face_data.relative_bounding_box.width), \
                            int(img_h*face_data.relative_bounding_box.height)
                break 

      
        if x+y+w+h > 0:
            print("Detected Face Points:", x,y,w,h)
            x = x - 10 
            y = y - 40 
            w = w + 20 
            h = h + 40 

            if x<0:
                x = 0
            if y<0:
                y = 0

            face_img = raw_img[y:y+h, x:x+w]
            print("Face Image:", face_img.shape)
            cv2.rectangle(raw_img, (x,y), (x+w, y+h), (255,0,0), 2) 

            face_crop = cv2.resize(face_img, (96,96)) 
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop) 
            face_crop = np.expand_dims(face_crop, axis=0) 

            conf = model.predict(face_crop)[0] 
            idx = np.argmax(conf)
            label = "{}: {:.2f}%".format(classes[idx], conf[idx] * 100)
            cv2.putText(raw_img, label, (x, y-50), FONTS, 1, TEXT_COLOR, 1)

          
            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) 
            face_resized = cv2.resize(face_img_gray, (48, 48)) 
            preds = emotion_model.predict(face_resized[np.newaxis, :, :, np.newaxis])
            pred = EMOTIONS_LIST[np.argmax(preds)]
            cv2.putText(raw_img, pred, (x, y-10), FONTS, 1, TEXT_COLOR, 1)         
            face_processed, raw_img = detect_eye_mouth_status(face_img, raw_img) 
            raw_img[y:y+h, x:x+w] = face_processed 
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False) 
            ageNet.setInput(blob) 
            agePreds = ageNet.forward() 
            age = ageList[agePreds[0].argmax()]
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
            cv2.putText(raw_img, "Age : {}".format(age), (x, y-30), FONTS, 1, TEXT_COLOR, 1)         
            mask = np.zeros_like(raw_img)           
            mask = cv2.ellipse(mask, (int((x+(w/2))), int(y+(h*0.7))),(w//3, w//3), 0, 0, 180, (255,255,255),thickness=-1) 
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)            
            only_beard_part = np.bitwise_and(raw_img, mask)         
            hsv_img = cv2.cvtColor(only_beard_part, cv2.COLOR_BGR2HSV)      
            low_black = np.array([94, 80, 2])
            high_black = np.array([126, 255, 255])
            MASK = cv2.inRange(hsv_img, low_black, high_black)           
            print(cv2.countNonZero(MASK))
            if cv2.countNonZero(MASK) < 110: 
                print("Beard Not Found")
                cv2.putText(raw_img, "Beard Not Found", (x, y+h), FONTS, 1, TEXT_COLOR, 1)
            elif cv2.countNonZero(MASK) < 150:
                print("Light Beard Found")
                cv2.putText(raw_img, "Light Beard Found", (x, y+h), FONTS, 1, TEXT_COLOR, 1)
            else:
                print("Beard Found")
                cv2.putText(raw_img, "Beard Found", (x, y+h), FONTS, 1, TEXT_COLOR, 1)
        cv2.imshow('frame', raw_img)
        cv2.waitKey(1)
        time_taken = time.time()-start_time
        fps = 1/time_taken
        print("FPS:", fps)


