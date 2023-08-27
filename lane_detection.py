import cv2
import numpy as np
import matplotlib.pyplot as plt

class Canny_edge_lane:
    def __init__(self, cap):
        self.cap = cap

    def grayscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, frame, kernel_size):
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX= 0)

    def canny(self, frame, low_t, high_t):
        return cv2.Canny(frame, low_t, high_t)

    def roi(self, frame):
        mask = np.zeros_like(frame)
        h, w = mask.shape
        a = w * 3/10
        b = h * 3/5
        vertices = np.array([[(0, h), (a, b), (w-a, b), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        roi_frame = cv2.bitwise_and(frame, mask)
        return roi_frame

    def open_vid(self):
        count = 0
        while True:
            # 동영상 파일 존재 여부(True/False)와 현재 프레임 이미지를 읽음
            retval, frame = self.cap.read()
            gray_frame = self.grayscale(frame)
            gaussian_blur_frame = self.gaussian_blur(gray_frame, kernel_size=5)
            canny_frame = self.canny(gaussian_blur_frame, low_t=50, high_t=200)
            roi_frame = self.roi(canny_frame)

            # 만약 동영상 파일이 존재하지 않으면 while 반복문 종료
            if retval == False:
                print('Vid is finished')
                break

            # 2개 이상의 키값 받기 위해 아래 코드 생성.
            key = cv2.waitKeyEx(10)
            # ESC 키 누르면 영상 종료
            if key == 27:
                break
            # Tab 키로 color, gray scale 영상 변환 출력
            elif key == 9:
                count += 1
                if count == 5:
                    count = 0

            if count == 0:
                cv2.imshow('road_driving', frame)  # 영상 원본 재생
            elif count == 1:
                cv2.imshow('road_driving', gray_frame) # 영상 Grayscale 재생
            elif count == 2:
                cv2.imshow('road_driving', gaussian_blur_frame)
            elif count == 3:
                cv2.imshow('road_driving', canny_frame)
            elif count == 4:
                cv2.imshow('road_driving', roi_frame)

        # 동영상 파일 닫고 모든창 종료
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    vid_name = 'highway.mp4'
    vid_path = 'C:/Users/강두인/Downloads/' + vid_name
    cap = cv2.VideoCapture(vid_path)

    lane_detection = Canny_edge_lane(cap)
    lane_detection.open_vid()


if __name__ == '__main__':
    main()
