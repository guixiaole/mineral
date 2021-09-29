import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2 as cv


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))


def result(image1, image2, wsize, isSobel, size=None):
    if isSobel:
        image1 = cv.Sobel(image1, ksize=size, dx=1, dy=1, ddepth=-1)
        image2 = cv.Sobel(image2, ksize=size, dx=1, dy=1, ddepth=-1)
    image1 = Image.fromarray(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    return compute_ssim(np.array(image1.resize((8, 8), Image.ANTIALIAS).convert('L'), 'f'),
                        np.array(image2.resize((8, 8), Image.ANTIALIAS).convert('L'), 'f'), win_size=wsize)


if __name__ == "__main__":
    videos_src_path = 'testVideo.mp4'
    cap = cv.VideoCapture(videos_src_path)
    cap.set(cv.CAP_PROP_POS_FRAMES, 60000)
    success = True
    prefram = None
    frame_count = 0
    while (success):
        success, frame = cap.read()
        if frame_count == 0:
            prefram = frame
        elif frame_count % 50 == 0:
            all_total = result(prefram, frame, 3, False) * result(prefram, frame, 3, True, 3) * result(prefram, frame,
                                                                                                       3, True, 7)
            print(all_total)
            if all_total < 0.98:
                cv.imshow("prefram", prefram)
                cv.imshow("frame", frame)
                cv.waitKey(-1)
                print("启动")
                break
            prefram = frame
        # cv.imshow('frame', frame)
        frame_count = frame_count + 1
    cap.release()
    # cv.destroyAllWindows()
    # image1 =cv.imread('63300.jpg')
    # image2 =cv.imread('63320.jpg')
    # image1 =cv.imread('testjpg/_63300.jpg')
    # image2 =cv.imread('testjpg/_63350.jpg')
    # print(image1.shape)
    # image1 = Image.open('63300.jpg')
    # image2 = Image.open('63320.jpg')
    # image1 = cv.Sobel(image1, ksize=3, dx=1, dy=1,ddepth=-1)
    # image2 = cv.Sobel(image2, ksize=3, dx=1, dy=1,ddepth=-1)
    # image1 = Image.fromarray(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    # image2 = Image.fromarray(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    # print(compute_ssim(np.array(image1.resize((8, 8), Image.ANTIALIAS).convert('L'), 'f'),np.array(image2.resize((8, 8), Image.ANTIALIAS).convert('L'), 'f')))
    # print(compute_ssim(image1,image2))
    # all_total=result(image1,image2,3,False)*result(image1,image2,3,True,3)*result(image1,image2,3,True,7)
    # if all_total<0.99:
    #     print("启动")
    #
    # print(all_total)
# 0.9989074837072285
# 0.9526762427039644
# 0.9399466501786743
