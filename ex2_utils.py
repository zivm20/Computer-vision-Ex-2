import math
import numpy as np
import cv2
from typing import Tuple
from ex1_utils import quantizeImage
import matplotlib.pyplot as plt
from sklearn import neighbors

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    out_signal = []
    for i in range(len(in_signal)+len(k_size)-1):
        subArr1 = in_signal[max(0,i-len(k_size)+1):min(i+1,len(in_signal))]
        #reverse kernel subarray
        subArr2 = k_size[max(0,i-len(in_signal)+1):min(i+1,len(k_size))][::-1]
        out_signal.append(np.dot(subArr1,subArr2))

    return np.array(out_signal)


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    img = cv2.copyMakeBorder(in_image, len(kernel), len(kernel), len(kernel[0]), len(kernel[0]),cv2.BORDER_REPLICATE)
    out = []
    norm = False
    #check if image is normalized or not
    if len(in_image[in_image<=1]) == in_image.shape[0]*in_image.shape[1]:
        norm = True
    
    for x in range(len(kernel)-len(kernel)//2, len(in_image)+len(kernel)-len(kernel)//2):
        row=[]
        for y in range(len(kernel[0])-len(kernel[0])//2, len(in_image[0])+len(kernel[0])-len(kernel[0])//2):
            #get sub matrix we are multiplying from the original image
            subMat = img[x:x+len(kernel), y:y+len(kernel[0])]
            value = np.sum(np.multiply(subMat,kernel))
            #if we aren't using normalized values, round the answer
            if norm == False:
                value = max(0,np.round(value))
            row.append(value)
        out.append(row)
    out = np.array(out)

    return np.array(out)


def convDerivative(in_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    ker = np.array([[1, 0, -1]])
    #get the derivatives of the image 
    X = conv2D(in_image,ker)
    Y = conv2D(in_image,ker.T)
    ori = np.arctan2(Y, X).astype(np.float64)
    mag = np.sqrt(X ** 2 + Y ** 2).astype(np.float64)
    return ori, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel_vector=np.array([1])
    #generate vector of binomial coefficients of size k_size
    while(len(kernel_vector)<k_size):
        kernel_vector = conv1D(kernel_vector,np.array([1,1]))
    kernel_vector = np.array([kernel_vector])
    #get our kernel from using matrix multiplication
    kernel = kernel_vector.T * kernel_vector
    return conv2D(in_image,kernel/np.sum(kernel))


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size,-1)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    return img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    smooth = blurImage1(img,15)
    #blur image and apply laplacian filter
    deriv = conv2D(smooth, np.array([[0, 1, 0], [1, -4, 1], [0, 1 , 0]]))
    
    out = []
    #zero crossing
    for i in range(deriv.shape[0]):
        row = []
        for j in range(deriv.shape[1]):
            if i>0 and i<deriv.shape[0]-1 and deriv[i-1][j]*deriv[i+1][j]<0:
                row.append(1)
            elif i>0 and deriv[i-1][j]*deriv[i][j]<0:
                row.append(1)
            elif j>0 and j<deriv.shape[0]-1 and deriv[i][j-1]*deriv[i][j+1]<0:
                row.append(1)
            elif j>0 and deriv[i][j-1]*deriv[i][j]<0:
                row.append(1)
            else:
                row.append(0)
        out.append(row)

    return np.array(out)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    newImg, errs = quantizeImage(img,50,50)
    newImg = newImg[-1]
    newImg = (255*blurImage1(newImg,11)).astype(np.uint8)
    #get 
    v = np.median(newImg)
    lower = int(max(0, 0.67 * v))
    upper = int(min(255, 1.33 * v))
    newImg = cv2.Canny(newImg,lower,upper)
    #plt.imshow(newImg)
    #plt.show()

    circle_vals = []
    degrees = np.arange(0,360,step=10)
    #precalculate some values for detecting circles
    for r in range(min_radius,max_radius+1):
        for angle in degrees:
            circle_vals.append((r,np.round(r*np.cos(angle*np.pi/180)), np.round(r*np.sin(angle*np.pi/180))))


    space = [[[0 for r in range(min_radius,max_radius+1)] for j in i]  for i in newImg]
    for y in range(len(newImg)):
        for x in range(len(newImg[y])):
            if newImg[y][x] > 0:
                for r, cos, sin in circle_vals:
                    b = int(y - sin)
                    a = int(x - cos)
                    #voting
                    if a >= 0 and a < len(space) and b >= 0 and b < len(space[a]):
                        space[a][b][r-min_radius] += 1

    circles = []
    circle_threshold=19
    space = np.array(space)
    for y in range(len(space)):
        for x in range(len(space[y])): 
            circle = None
            confidence = float('inf')
            #get the outermost circle with the highest confidence
            for r in range(len(space[y][x])):
                if space[y][x][r] > circle_threshold and confidence > space[y][x][r]:
                    circle = (y,x,r+min_radius)
                    confidence = space[y][x][r]
            if circle != None:
                circles.append(circle)
    print(circles)
    print(space.shape)
    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    out = []
    #pad image
    img = cv2.copyMakeBorder(in_image, k_size, k_size, k_size, k_size,cv2.BORDER_REPLICATE)
    for i in range(k_size, img.shape[0]-k_size):
        row = []
        for j in range(k_size, img.shape[1]-k_size):
            #apply bilateral filter function for each pixel 
            row.append(bilateral_filter(img,k_size,sigma_space,sigma_color,i,j))
        out.append(row)
    out = np.array(out)
    out2 = cv2.bilateralFilter(in_image,k_size,sigmaColor=sigma_color,sigmaSpace=sigma_space)
    return out2, out


def bilateral_filter(in_image,k_size,sigma_space,sigma_color,i,j):
    #pixel bilateral function
    pivot = in_image[i][j]
    neighbors = in_image[i-k_size:i+k_size+1,j-k_size:j+k_size+1]
    diff = pivot - neighbors
    diff_gau = np.exp(-np.power(diff, 2)/(2*sigma_space))
    gaus = cv2.getGaussianKernel(2*k_size+1,sigma_color)
    gaus = gaus.dot(gaus.T)
    combo = gaus*diff_gau
    value = np.sum(combo*neighbors/np.sum(combo))
    return np.round(value)
            