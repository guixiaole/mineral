import  cv2 as cv

cap = cv.VideoCapture("F:/BaiduYunDownload/北胶机尾_20D47898_1541063615_1.mp4")
flag = 1
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# 视频的高度
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
fps = cap.get(cv.CAP_PROP_FPS)
out = cv.VideoWriter('F:/BaiduYunDownload/output1.avi', fourcc, fps, (width, height))
out1 = cv.VideoWriter('F:/BaiduYunDownload/output2.avi', fourcc, fps, (width, height))
cap.set(cv.CAP_PROP_POS_FRAMES, 20)
temp_flag = 1
while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        if flag>1000  and temp_flag ==1:
            temp_flag = 0
            flag = 500
            cap.set(cv.CAP_PROP_POS_FRAMES, flag)
            while flag< 1000:
                ret1,frame1 = cap.read()
                out.write(frame1)
                flag+=1

    else:
        break
    if flag>1003 and flag<1500:
        out1.write(frame)
    if flag>1500:
        break
    flag+=1

cap.release()
out.release()
out1.release()