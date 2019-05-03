import cv2
import dlib
import numpy as np


PREDICTOR_PATH = "../model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cascade_path = "../haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(img):
    rects = cascade.detectMultiScale(img,1.3,5)
    #detectMultiScale(画像path,縮小量 (誤検出の調整),最低限矩形数(制度をあげるため))
    (x,y,w,h) = rects[0]
    rect = dlib.rectangle(x,y,x+w,y+h)
    #検出するもののレクタングルの大きさ
    return np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])

def annotate_landmarks(img,landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0,0],point[0,1])
        cv2.putText(img,str(idx),pos,
            fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale = 0.4,
            color = (255,0,0))
            #RGBではなくBGRなので注意
        cv2.circle(img,pos,2,color =(255,0,255))
    return img

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while(True):
        try:
            ret,img = cap.read()

            height = round(img.shape[0]/2)
            width = round(img.shape[1]/2)

            img = cv2.resize(img,(width,height))
            img = annotate_landmarks(img,get_landmarks(img))

        except:
            print("erorr!")

        cv2.imshow('Result',img)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
