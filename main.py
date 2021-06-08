import sys
from time import time

import cv2 as cv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

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
        classes = "mineral"
    return classes, boxes, confidences


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

        if self.videoName is not None:
            cap = cv.VideoCapture(self.videoName)
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            # 视频的高度
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

            # 视频的帧率  视频的编码  定义视频输出

            # fps = cap.get(cv.CAP_PROP_FPS)
            fps = 20
            # out = cv.VideoWriter("E:/mineral/out" + ".mp4", fourcc, fps, (width, height))
            fourcc = cv.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
            out = cv.VideoWriter('E:/mineral/output1.avi', fourcc, fps, (width, height))
            flag = 1
            while cap.isOpened():
                ret, frame = cap.read()
                flag += 1

                if ret:
                    classes, boxes, confidences = yolov4_detect(frame)
                    print(classes, boxes, confidences)
                    if len(boxes) > 0:
                        for i in range(len(boxes)):
                            var = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                            cv.rectangle(frame, (var[0], var[1]), (var[0] + var[2], var[1] + var[3]), (0, 255, 0), 3)
                    out.write(frame)
                    print(flag)
                else:
                    break
            cap.release()
            out.release()


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = VideoCapture()
    form.show()
    app.exec_()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
