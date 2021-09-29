import sys
from time import time

import cv2 as cv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import all
from mineral import Ui_MainWindow


def yolov4_detect(frame):
    net = cv.dnn_DetectionModel('yolo-obj.cfg', 'yolo-obj_last.weights')
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)

    # frame = cv.imread('images/test4.jpg')

    with open('obj.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.8, nmsThreshold=0.5)

    # for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    #     label = '%.2f' % confidence
    #     label = '%s: %s' % (names[classId], label)
    #     labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #     left, top, width, height = box
    #     top = max(top, labelSize[1])
    #     cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
    #     cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
    #                  cv.FILLED)
    #     cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    #
    # cv.imshow('out', frame)
    # cv.waitKey(0)
    if len(classes) > 0:
        classes = "coke"
    return classes, boxes, confidences


def get_similar(last_frame, now_frame):
    return all.detectionSSIM(last_frame,now_frame)

def get_coke(frame):
    return all.detectionCoke(frame)


def judge_coke(is_coke):
    """
    判断是不是处于有煤与无煤之间的转换
    返回为0 则不判断处于什么情况
    返回为1 则判断为处于有煤向无煤进行转换
    返回为-1 则判断为无煤向有煤进行转换
    :param is_coke:
    :return:
    """
    if len(is_coke) < 10:
        return 0
    # 假设准直在5左右，
    if sum(is_coke) == 5:
        if sum(is_coke[:5]) >= 3:
            return 1
        else:
            return -1
    return 0


similar_yuzhi = 9
similar_half = 4


def judge_similar(similar):
    if len(similar) < 10:
        return 0
    if sum(similar) < similar_yuzhi:
        if sum(similar[:5]) < similar_half:
            return 1
        elif sum(similar[5:-1]):
            return -1
    return 0


def judge_master_coke(master_coke):
    judge_sum = sum(master_coke)
    length = len(master_coke)
    if judge_sum < length * 0.4 or judge_sum > length * 0.6:
        return -1
    else:
        return 1


def judge_master_similar(master_similar):
    judge_sum = sum(master_similar)
    length = len(master_similar)
    if judge_sum < length * 0.90:
        return 1
    else:
        return -1


class VideoCapture(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.videoName = "E:/mineral/1.mp4"
        self.videoName = ""

    def loadVideoFile(self):

        self.videoName, videoType = QFileDialog.getOpenFileName(self,
                                                                "打开视频",
                                                                "",
                                                                " *.mp4;;*.avi;;All Files (*)"
                                                                )

        print(self.videoName)

    def DownloadVideo(self):
        self.saveVideo()

    def saveVideo(self, save_name_video=1):
        if self.videoName is not None:
            cap = cv.VideoCapture(self.videoName)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            # 视频的高度
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            # 相似性与有煤无煤之间的转换
            similar = []
            is_coke = []
            res_similar = []
            res_coke = []
            # 视频的帧率  视频的编码  定义视频输出

            # fps = cap.get(cv.CAP_PROP_FPS)
            fps = 20
            # out = cv.VideoWriter("E:/mineral/out" + ".mp4", fourcc, fps, (width, height))
            fourcc = cv.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            # out = cv.VideoWriter('E:/mineral/output1.avi', fourcc, fps, (width, height))
            flag = 0

            while cap.isOpened():
                ret, frame = cap.read()
                flag += 1
                last_frame = None
                if ret:
                    if flag == 0:
                        last_frame = frame
                        flag += 1
                        continue
                    # 第一帧不进行检测，其余的进行检测
                    singal_similar = get_similar(last_frame, frame)
                    singal_coke = get_coke(frame)
                    if len(similar) > 10:
                        similar.pop(0)
                    similar.append(singal_similar)
                    if len(is_coke) > 10:
                        is_coke.pop(0)
                    is_coke.append(singal_coke)
                    last_frame = frame
                    flag += 1
                    coke_flag = 0
                    similar_flag = 0
                    if judge_coke(is_coke) == 1:
                        coke_flag = 1
                    elif judge_coke(is_coke) == -1:
                        coke_flag = 2
                    if judge_similar(similar) == 1:
                        similar_flag = 1
                    elif judge_similar(similar) == -1:
                        similar_flag = 2
                    # 假设已经做了启动检测，则可能需要向前跳20多秒，再来进行检测
                    if coke_flag <= 0 and similar_flag <= 0:
                        continue
                    temp_flag = flag
                    flag = flag - 50 * 10
                    cap.set(cv.CAP_PROP_POS_FRAMES, flag)
                    master_coke_detect = []
                    master_similar_detect = []
                    while flag < temp_flag:
                        ret1, frame1 = cap.read()
                        if coke_flag > 0:
                            master_coke_detect.append(get_coke(frame1))
                        if similar_flag > 0:
                            master_similar_detect.append(get_similar(last_frame, frame1))
                        last_frame = frame1
                        flag += 1
                    coke_detect_result = judge_master_coke(master_coke_detect)
                    similar_detect_result = judge_master_similar(master_similar_detect)
                    if coke_detect_result >= 1 or similar_detect_result >= 1:
                        flag = flag - 50 * 10
                        out1 = cv.VideoWriter('F:/BaiduYunDownload/output' + str(save_name_video) + '.avi', fourcc, fps,
                                              (width, height))
                        save_name_video += 1
                        cap.set(cv.CAP_PROP_POS_FRAMES, flag)
                        while flag < temp_flag:
                            ret1, frame1 = cap.read()
                            out1.write(frame1)
                            flag+=1
                        out1.release()
                else:
                    break
            cap.release()

    def detectCokeVideo(self):
        if self.videoName is not None:
            cap = cv.VideoCapture(self.videoName)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            # 视频的高度
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            # 相似性与有煤无煤之间的转换
            is_coke = []
            res_coke = []
            # 视频的帧率  视频的编码  定义视频输出

            # fps = cap.get(cv.CAP_PROP_FPS)
            fps = 20
            # out = cv.VideoWriter("E:/mineral/out" + ".mp4", fourcc, fps, (width, height))
            fourcc = cv.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            # out = cv.VideoWriter('E:/mineral/output1.avi', fourcc, fps, (width, height))
            flag = 0
            save_name_video = 1
            while cap.isOpened():
                ret, frame = cap.read()
                flag += 1
                last_frame = None
                if ret:
                    if flag == 0:
                        last_frame = frame
                        flag += 1
                        continue
                    # 第一帧不进行检测，其余的进行检测
                    singal_coke = get_coke(frame)
                    if len(is_coke) > 10:
                        is_coke.pop(0)
                    is_coke.append(singal_coke)
                    last_frame = frame
                    flag += 1
                    coke_flag = 0

                    if judge_coke(is_coke) == 1:
                        coke_flag = 1
                    elif judge_coke(is_coke) == -1:
                        coke_flag = 2
                    # 假设已经做了启动检测，则可能需要向前跳20多秒，再来进行检测
                    if coke_flag <= 0:
                        continue
                    temp_flag = flag
                    flag = flag - 50 * 10
                    cap.set(cv.CAP_PROP_POS_FRAMES, flag)
                    master_coke_detect = []
                    while flag < temp_flag:
                        ret1, frame1 = cap.read()
                        if coke_flag > 0:
                            master_coke_detect.append(get_coke(frame1))
                        last_frame = frame1
                        flag += 1
                    coke_detect_result = judge_master_coke(master_coke_detect)
                    if coke_detect_result >= 1:
                        flag = flag - 50 * 10
                        out1 = cv.VideoWriter('F:/BaiduYunDownload/cokeDetect' + str(save_name_video) + '.avi', fourcc, fps,
                                              (width, height))
                        save_name_video += 1
                        cap.set(cv.CAP_PROP_POS_FRAMES, flag)
                        while flag < temp_flag:
                            ret1, frame1 = cap.read()
                            out1.write(frame1)
                            flag+=1
                        out1.release()
                        msg_box = QMessageBox(QMessageBox.Warning, '提示', '文件已经存储在F盘中')
                        msg_box.exec_()
                else:
                    break
            cap.release()

    def detectSimilarVideo(self):
        if self.videoName is not None:
            cap = cv.VideoCapture(self.videoName)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            # 视频的高度
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            # 相似性与有煤无煤之间的转换
            similar = []
            res_similar = []
            # 视频的帧率  视频的编码  定义视频输出
            # fps = cap.get(cv.CAP_PROP_FPS)
            fps = 20
            fourcc = cv.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            flag = 0
            save_name_video = 1
            while cap.isOpened():
                ret, frame = cap.read()
                flag += 1
                last_frame = None
                if ret:
                    if flag == 0:
                        last_frame = frame
                        flag += 1
                        continue
                    # 第一帧不进行检测，其余的进行检测
                    singal_similar = get_similar(last_frame, frame)
                    if len(similar) > 10:
                        similar.pop(0)
                    similar.append(singal_similar)
                    last_frame = frame
                    flag += 1
                    similar_flag = 0
                    if judge_similar(similar) == 1:
                        similar_flag = 1
                    elif judge_similar(similar) == -1:
                        similar_flag = 2
                    # 假设已经做了启动检测，则可能需要向前跳20多秒，再来进行检测
                    if similar_flag <= 0:
                        continue
                    temp_flag = flag
                    flag = flag - 50 * 10
                    cap.set(cv.CAP_PROP_POS_FRAMES, flag)
                    master_similar_detect = []
                    while flag < temp_flag:
                        ret1, frame1 = cap.read()
                        if similar_flag > 0:
                            master_similar_detect.append(get_similar(last_frame, frame1))
                        last_frame = frame1
                        flag += 1
                    similar_detect_result = judge_master_similar(master_similar_detect)
                    if similar_detect_result >= 1:
                        flag = flag - 50 * 10
                        out1 = cv.VideoWriter('F:/BaiduYunDownload/similarDetect' + str(save_name_video) + '.avi', fourcc, fps,
                                              (width, height))
                        save_name_video += 1
                        cap.set(cv.CAP_PROP_POS_FRAMES, flag)
                        while flag < temp_flag:
                            ret1, frame1 = cap.read()
                            out1.write(frame1)
                            flag+=1
                        out1.release()
                        msg_box = QMessageBox(QMessageBox.Warning, '提示', '文件已经存储在F盘中')
                        msg_box.exec_()
                else:
                    break
            cap.release()


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = VideoCapture()
    form.show()
    app.exec_()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
