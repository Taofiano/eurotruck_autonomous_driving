import cv2
import numpy as np
import matplotlib.pyplot as plt


# step2.동영상 파일 열기
vid_name = 'highway.mp4'
vid_path = 'C:/Users/강두인/Downloads/' + vid_name
cap = cv2.VideoCapture(vid_path)

# step3.무한 반복
# while cap.isOpened(): 아래줄과 같은 코드
while True:
    # 동영상 파일 존재 여부(True/False)와 현재 프레임 이미지를 읽음
    retval, frame = cap.read()

    # 만약 동영상 파일이 존재하지 않으면 while 반복문 종료
    if retval == False:
        print('Vid is finished')
        break

    # 2개 이상의 키값 받기 위해 아래 코드 생성.
    key = cv2.waitKeyEx(10)
    if key == 27:
        break
    elif key == ord('g'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('road_driving', gray)
    cv2.imshow('road_driving', frame)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if cv2.waitKey(10) == ord('g'):
        #     cv2.imshow('road_driving', gray)
        # if cv2.waitKey(10) == 27:
        #     break
        # ESC 키가 입력되면 while 반복문 종료


    # # 'frame'이란 창 이름으로 현g재 프레임 출력
    # cv2.imshow('road_driving', frame)
    #
    # # ESC 키가 입력되면 while 반복문 종료
    # if cv2.waitKey(10) == 27:
    #     break

# step4.동영상 파일 닫고 모든창 종료
cap.release()
cv2.destroyAllWindows()
