import onnxruntime as ort
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2 as cv

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    return np.mean(np.mean(ssim_map))
def result(image1,image2,wsize,isSobel,size=None):
    if isSobel:
        image1 = cv.Sobel(image1, ksize=size, dx=1, dy=1,ddepth=-1)
        image2 = cv.Sobel(image2, ksize=size, dx=1, dy=1,ddepth=-1)
    image1 = Image.fromarray(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    return compute_ssim(np.array(image1.resize((8, 8), Image.ANTIALIAS).convert('L'), 'f'),np.array(image2.resize((8, 8), Image.ANTIALIAS).convert('L'), 'f'),win_size=wsize)
# def detectionCoke(img):
#     img=cv.resize(img,dsize=(224,224))
#     img=np.divide(img,255)
#     img=img[np.newaxis,:].transpose(0,3,1,2)
#     _,outputs = ort_session.run(None, {'input': img.astype(np.float32)})
#     # class_names=['Coke','NoCoke']
#     indices=np.argmax(outputs,1)
#     # _, indices = torch.max(outputs, 1)
#     # percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
#     # perc = percentage[int(indices)].item()
#     return indices[0]^1
def detectionSSIM(prefram,frame):
    all_total = result(prefram, frame, 3, False) * result(prefram, frame, 3, True, 5)
    return all_total
