import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def onChange 및 'canny'라는 창의 주석들 : 
Canny edge 검출 시 low_T, high_T 트랙바 활용해 실험.
'''

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
    def hough(self, frame, min_deg, max_deg):
        h, w = frame.shape
        lines = cv2.HoughLinesP(frame, rho=1, theta=np.pi/180, threshold=50, minLineLength=10, maxLineGap=30)
        line_img = np.zeros((h, w, 3), dtype=np.uint8)
        lines, deg = self.lines_wanted(lines, min_deg, max_deg)
        print(lines, deg)
        for x1, y1, x2, y2 in lines:
        # for line in lines:
        # for x1, y1, x2, y2 in line:
            # color : [B, G, R]
            cv2.line(line_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=2)
        return line_img

    def lines_wanted(self, lines, min_deg=130, max_deg=180):
        lines = np.squeeze(lines)
        theta_deg = np.abs(np.rad2deg(np.arctan2(lines[:, 1] - lines[:, 3], lines[:, 0] - lines[:, 2])))
        lines = lines[np.abs(theta_deg) < max_deg]
        theta_deg = theta_deg[np.abs(theta_deg) < max_deg]
        lines = lines[np.abs(theta_deg) > min_deg]
        theta_deg = theta_deg[np.abs(theta_deg) > min_deg]

        # np.append(lines_new, lines_new[np.abs(theta_deg) > 30])
        # np.append(lines_new, lines[np.abs(theta_deg) > 110])
        # np.append(lines_new, lines_new[np.abs(theta_deg) > 30])
        return lines, theta_deg
    def special_lines(self, frame, line_classify_count=0):
        h, w = frame.shape
        line_img = np.zeros((h, w, 3), dtype=np.uint8)
        lines = cv2.HoughLinesP(frame, rho=1, theta=np.pi / 180, threshold=50, minLineLength=10, maxLineGap=30)
        lines, theta_deg = self.lines_wanted(lines, min_deg=130, max_deg=180)

        line_arr_1 = np.empty((1, 4), dtype=float)
        line_arr_2 = np.empty((1, 4), dtype=float)

        for index in range(theta_deg.size):

            if index <= theta_deg.size - 1:
                if theta_deg[index] - theta_deg[index + 1] < 7:
                    if line_classify_count == 0:
                        np.append(line_arr_1, lines[index], axis=0)
                        print(line_arr_1)
                    elif line_classify_count == 1:
                        np.append(line_arr_2, lines[index], axis=0)

                elif theta_deg[index] - theta_deg[index + 1] >= 7:
                    line_classify_count += 1
                    np.append(line_arr_1, theta_deg[index], axis=0)
                    break
        print(line_arr_1)
        line1_x1 = line_arr_1(axis=0) / theta_deg.size
        line1_y1 = line_arr_1(axis=1) / theta_deg.size
        line1_x2 = line_arr_1(axis=2) / theta_deg.size
        line1_y2 = line_arr_1(axis=3) / theta_deg.size

        line2_x1 = line_arr_2(axis=0) / theta_deg.size
        line2_y1 = line_arr_2(axis=1) / theta_deg.size
        line2_x2 = line_arr_2(axis=2) / theta_deg.size
        line2_y2 = line_arr_2(axis=3) / theta_deg.size

        # line1_x1 = np.sum(line_list_1[:, 0]) / theta_deg.size
        # line1_y1 = np.sum(line_list_1[:, 1]) / theta_deg.size
        # line1_x2 = np.sum(line_list_1[:, 2]) / theta_deg.size
        # line1_y2 = np.sum(line_list_1[:, 3]) / theta_deg.size
        #
        # line2_x1 = np.sum(line_list_2[:, 0]) / theta_deg.size
        # line2_y1 = np.sum(line_list_2[:, 1]) / theta_deg.size
        # line2_x2 = np.sum(line_list_2[:, 2]) / theta_deg.size
        # line2_y2 = np.sum(line_list_2[:, 3]) / theta_deg.size

        cv2.line(line_img, (line1_x1, line1_y1), (line1_x2, line1_y2), color=[0, 0, 255], thickness=3)
        cv2.line(line_img, (line2_x1, line2_y1), (line2_x2, line2_y2), color=[0, 255, 0], thickness=3)

        return line_img

    # Trackbar 생성을 위한 pass 함수
    # def onChange(self, x):
    #     pass
    def open_vid(self):
        count = 0

        # canny 라인을 위한 Threshold 실험값 측정 트랙바
        # cv2.namedWindow('canny')
        # cv2.createTrackbar('low_T', 'canny', 0, 255, self.onChange)
        # cv2.createTrackbar('high_T', 'canny', 0, 255, self.onChange)

        # hough 변환, 차선 detection 최소, 최대 각도 실험값 측정 트랙바
        # cv2.namedWindow('deg_cont')
        # cv2.createTrackbar('min_deg', 'deg_cont', 0, 180, self.onChange)
        # cv2.createTrackbar('max_deg', 'deg_cont', 0, 180, self.onChange)

        while True:
            # 동영상 파일 존재 여부(True/False)와 현재 프레임 이미지를 읽음
            retval, frame = self.cap.read()
            gray_frame = self.grayscale(frame)
            # 가우시안 블러 kernel_size:3 cause: Sobel kernel to be used internally
            gaussian_blur_frame = self.gaussian_blur(gray_frame, kernel_size=3)
            canny_frame = self.canny(gaussian_blur_frame, low_t=130, high_t=200)
            roi_frame = self.roi(canny_frame)
            # hough_frame = self.hough(roi_frame)

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
                if count == 7:
                    count = 0

            if count == 0:
                cv2.imshow('road_driving', frame)  # 영상 원본 재생
            elif count == 1:
                cv2.imshow('road_driving', gray_frame) # 영상 Grayscale 재생
            elif count == 2:
                cv2.imshow('road_driving', gaussian_blur_frame)
            elif count == 3:
                # low_t = cv2.getTrackbarPos('low_T', 'canny')
                # high_t = cv2.getTrackbarPos('high_T', 'canny')
                # canny_frame = self.canny(gaussian_blur_frame, low_t, high_t)
                cv2.imshow('road_driving', canny_frame)
            elif count == 4:
                cv2.imshow('road_driving', roi_frame)
            elif count == 5:
                # min_deg = cv2.getTrackbarPos('min_deg', 'deg_cont')
                # max_deg = cv2.getTrackbarPos('max_deg', 'deg_cont')
                hough_frame = self.hough(roi_frame, min_deg=130, max_deg=180)

                # cv2.imshow('deg_cont', hough_frame)
                cv2.imshow('road_driving', hough_frame)
            elif count == 6:
                special_frame = self.special_lines(roi_frame)

                # cv2.imshow('deg_cont', hough_frame)
                cv2.imshow('road_driving', special_frame)

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
