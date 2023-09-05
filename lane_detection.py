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
        return cv2.Canny(frame, low_t, high_t, L2gradient=True)

    def roi(self, frame):
        mask = np.zeros_like(frame)
        h, w = mask.shape
        print(h, w)
        a = w * 3/10
        b = h * 3/5
        c = w / 2
        vertices1 = np.array([[(0, h), (a, b), (w - a, b), (w, h)]], dtype=np.int32)
        vertices2 = np.array([[(a, h), (a, b), (w - a, b), (w - a, h)]], dtype=np.int32)
        mask = cv2.fillPoly(mask, vertices1, 255)
        # mask = cv2.fillPoly(mask, vertices2, 255)
        roi_frame = cv2.bitwise_and(frame, mask)
        return roi_frame
    def hough(self, frame, min_deg, max_deg):
        h, w = frame.shape
        lines = cv2.HoughLinesP(frame, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)
        line_img = np.zeros((h, w, 3), dtype=np.uint8)
        lines, deg = self.lines_wanted(lines, min_deg, max_deg)
        print(lines)
        print(deg)
        for x1, y1, x2, y2 in lines:
        # for line in lines:
        # for x1, y1, x2, y2 in line:
            # color : [B, G, R]
            cv2.line(line_img, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=2)
        return line_img

    def lines_wanted(self, lines, min_deg=-50, max_deg=50):
        lines = np.squeeze(lines)
        theta_deg = np.rad2deg(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))
        # print(lines)
        # print(theta_deg)

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
        lines = cv2.HoughLinesP(frame, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)
        lines, theta_deg = self.lines_wanted(lines, min_deg=-50, max_deg=50)

        line_pos_arr = np.empty((0, 4), float)
        line_neg_arr = np.empty((0, 4), float)

        deg_pos_mean = 0
        deg_neg_mean = 0

        deg_pos_count = 0
        deg_neg_count = 0

        for index in range(theta_deg.size):
            if index < (theta_deg.size - 1):
                if theta_deg[index] > 0:
                    line_pos_arr = np.vstack((line_pos_arr, lines[index]))
                    deg_pos_count += 1
                    if deg_pos_count == 1:
                        deg_pos_mean += theta_deg[index]
                    else:
                        deg_pos_mean = (deg_pos_mean + theta_deg[index]) / 2
                    if abs(deg_pos_mean - theta_deg[index + 1]) < 10:
                        line_pos_arr = np.vstack((line_pos_arr, lines[index + 1]))
                    # else:
                    #     line_pos_arr = np.vstack((line_pos_arr, lines[index]))

                elif theta_deg[index] < 0:
                    line_neg_arr = np.vstack((line_neg_arr, lines[index]))
                    deg_neg_count += 1
                    if deg_neg_count == 1:
                        deg_neg_mean += theta_deg[index]
                    else:
                        deg_neg_mean = (deg_neg_mean + theta_deg[index]) / 2
                    if abs(deg_neg_mean - theta_deg[index + 1]) < 10:
                        line_neg_arr = np.vstack((line_neg_arr, lines[index + 1]))
                    # else:
                    #     line_neg_arr = np.vstack((line_neg_arr, lines[index]))
            # 마지막 인덱스 처리
            # elif index == theta_deg.size:
            #     index = -1

        line_pos_arr = np.unique(line_pos_arr, axis=0)
        line_neg_arr = np.unique(line_neg_arr, axis=0)
        pos_x1, pos_y1, pos_x2, pos_y2 = line_pos_arr.sum(axis=0)
        neg_x1, neg_y1, neg_x2, neg_y2 = line_neg_arr.sum(axis=0)
        cv2.line(line_img, (int(pos_x1), int(pos_y1)), (int(pos_x2), int(pos_y2)), color=[255, 0, 0],
                 thickness=3)
        cv2.line(line_img, (int(neg_x1), int(neg_y1)), (int(neg_x2), int(neg_y2)), color=[0, 0, 255],
                 thickness=3)
        return line_img

    # def special_lines(self, frame, line_classify_count=0):
    #     h, w = frame.shape
    #     line_img = np.zeros((h, w, 3), dtype=np.uint8)
    #     lines = cv2.HoughLinesP(frame, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=100)
    #     lines, theta_deg = self.lines_wanted(lines, min_deg=30, max_deg=140)
    #
    #     line_pos_arr = []
    #     line_neg_arr = []
    #
    #     for index in range(theta_deg.size):
    #         if index < (theta_deg.size - 1):
    #             if theta_deg[index] > 0:
    #                 if abs(theta_deg[index] - theta_deg[index + 1]) < 10:
    #                     line_pos_arr.append(lines[index])
    #
    #             elif theta_deg[index] < 0:
    #                 if abs(theta_deg[index] - theta_deg[index + 1]) < 10:
    #                     line_neg_arr.append(lines[index])
    #
    #         line_pos_arr = np.array(line_pos_arr)
    #         line_neg_arr = np.array(line_neg_arr)
    #
    #         if line_pos_arr.size > 0:
    #             line1_x1 = np.mean(line_pos_arr[:, 0])
    #             line1_y1 = np.mean(line_pos_arr[:, 1])
    #             line1_x2 = np.mean(line_pos_arr[:, 2])
    #             line1_y2 = np.mean(line_pos_arr[:, 3])
    #
    #             cv2.line(line_img, (int(line1_x1), int(line1_y1)), (int(line1_x2), int(line1_y2)), color=[0, 0, 255],
    #                      thickness=3)
    #
    #         if line_neg_arr.size > 0:
    #             line2_x1 = np.mean(line_neg_arr[:, 0])
    #             line2_y1 = np.mean(line_neg_arr[:, 1])
    #             line2_x2 = np.mean(line_neg_arr[:, 2])
    #             line2_y2 = np.mean(line_neg_arr[:, 3])
    #
    #             cv2.line(line_img, (int(line2_x1), int(line2_y1)), (int(line2_x2), int(line2_y2)), color=[0, 255, 0],
    #                      thickness=3)
    #     return line_img


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
                hough_frame = self.hough(roi_frame, min_deg=20, max_deg=140)

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
    vid_name = 'highway.mp4' # 1280x720 & 12s
    # vid_name = 'highway2.mp4' # 3840x2160
    # vid_name = 'highway3.mp4' # 1280x720 & 14s
    vid_path = 'C:/Users/강두인/Downloads/' + vid_name
    cap = cv2.VideoCapture(vid_path)

    lane_detection = Canny_edge_lane(cap)
    lane_detection.open_vid()


if __name__ == '__main__':
    main()
