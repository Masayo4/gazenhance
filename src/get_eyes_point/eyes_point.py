import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../model/shape_predictor_68_face_landmarks.dat")


def get_face_dot(frame):
    dets = detector(frame[:,:,::-1])

    if len(dets) >0:
        parts = predictor(frame,dets[0]).parts()

        output_frame = frame *0.5
        for i in parts:
            cv2.circle(output_frame,(i.x,i.y),3,(255,0,0),-1)
            cv2.imshow("dot_face",output_frame)
            """
            68点の分類
            0~16 輪郭, 17~21 左眉　22~26 右眉　27~30 鼻筋 31~35 鼻
            36~41 左目(36が9時の位置,そこから時計回り)　42~47 左目(42が9時の位置,そこから時計回り)
            48~59 口(外側) 49が9時の位置,そこから時計回り
            60~67 口(内側) 60が9時の位置,そこから時計回り
            詳細は, ~/gazenhance/dlibPointNum.pngを参照
            """
    return parts

def pupil_detector(img,parts,left=True):
    if left:
        eyes =[
        parts[36],
        min(parts[37],parts[38],key=lambda x:x.y),#min(指定変数1,指定変数2,key) 最小値を算出する関数,keyはソートするときの順序づけのときに使用する lambdaは変数,この場合は小さい方のy座標を取りに行く
        max(parts[40],parts[41],key=lambda x:x.y),#max(指定変数1,指定変数2,key) 最大値を算出する関数 max(list)とするとlistの中にある最大値の取得が可能になる
        parts[39]
        ]

    else:
        eyes = [parts[42],
        min(parts[43],parts[44],key=lambda x:x.y),
        max(parts[46],parts[47],key=lambda x:x.y),
        parts[45]
        ]
        
    org_x = eyes[0].x
    org_y = eyes[1].y
    #瞳の中心を出すために取って来た座標の値を入れておく
    if eye_close(org_y,eyes[2].y):
        return None

    #目玉をトリミングする処理
    eye = img[org_y:eyes[2].y,org_x:eyes[-1].x]
    _, eye = cv2.threshold(cv2.cvtColor(eye,cv2.COLOR_RGB2GRAY),30,255,cv2.THRESH_BINARY_INV) #2値化する
    """
    cv2.cvtColor(画像,変換の仕方) COLOR_RGB2GRAYをするとBGRからRGBへの変換になる(openCVはBGRで値が返されるため)
    cv2.threshold(画像,閾値,最大値,方式)
    THRESH_BINARY_INVは最小値,最大値の順番に返ってくる 最大値は THRESH_BINARY,THRESH_BINARY_INVのときに使用
    """
    center = get_center(eye)
    if center:
        return center[0] + org_x, center[1] + org_y
    return center


def get_center(gray_img):
    moments = cv2.moments(gray_img,False)
    #cv2.moments(画像,bool) 重心や面積を求めてくれる便利な関数 trueにするとピクセル0を1として計算してくれる(trueは均一な密度のときしか使えない)
    try:
        return int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])
    except:
        return None

def eye_close(y0,y1):
    if abs(y0-y1) < 12:#y座標の差分計算(ケースによって変更する必要あり)
        return True
    return False

def draw_eye_center(img,parts,eye):
    if eye[0]:
        cv2.circle(img,eye[0],3,(255,255,0),-1)
    if eye[1]:
        cv2.circle(img,eye[1],3,(255,255,0),-1)
    #for i in parts:
        #cv2.circle(img,(i.x,i.y),3,(255,0,255),-1)

    cv2.imshow("dot_face",img)



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret,frame = cap.read()

        dets = detector(frame[:,:,::-1])

        if len(dets) >0:
            parts = predictor(frame,dets[0]).parts()

            output_frame = frame *0.5
            for i in parts:
                cv2.circle(output_frame,(i.x,i.y),3,(255,0,0),-1)
                cv2.imshow("dot_face",output_frame)
                """
                68点の分類
                0~16 輪郭, 17~21 左眉　22~26 右眉　27~30 鼻筋 31~35 鼻
                36~41 左目(36が9時の位置,そこから時計回り)　42~47 左目(42が9時の位置,そこから時計回り)
                48~59 口(外側) 49が9時の位置,そこから時計回り
                60~67 口(内側) 60が9時の位置,そこから時計回り
                詳細は, ~/gazenhance/dlibPointNum.pngを参照
                """
            cv2.namedWindow("dot_face", cv2.WINDOW_NORMAL)#windowの大きさを調整可能にする
            left_eye = pupil_detector(frame,parts)
            right_eye = pupil_detector(frame,parts,False)
            draw_eye_center(frame,parts,(left_eye,right_eye))
            cv2.moveWindow('dot_face', 20, 20) #windowの表示位置
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

"""
参考
https://cppx.hatenablog.com/entry/2017/12/25/231121
"""
