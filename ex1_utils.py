"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 209904606


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        img = cv2.cvtColor(cv2.imread(filename,1), cv2.COLOR_BGR2RGB)
    except:
        return np.zeros( (256,256,3))
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img/255



def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    
    img = imReadAndConvert(filename,representation)
    if representation == 1:
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    #ensure image has 3 channels
    try:
        ensure3ChannelImgInput(imgRGB)
    except Exception as e:
        print(e)
        return imgRGB

    #YIQ conversion matrix
    YIQ_mat = np.array([[ 0.299, 0.587, 0.114],
                        [ 0.596, -0.275, -0.321],
                        [ 0.212, -0.523, 0.311]])
    
    #turn our NxMx3 matrix to (N*M)x3 matrix
    img_vals = imgRGB.reshape((imgRGB.shape[0] * imgRGB.shape[1], 3))

    #matrix multiplication to convert each RGB triplet into YIQ
    img_vals = np.matmul(img_vals,YIQ_mat)

    #reshape our matrix back
    return img_vals.reshape(imgRGB.shape).astype(float)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    #ensure image has 3 channels
    try:
        ensure3ChannelImgInput(imgYIQ)
    except Exception as e:
        print(e)
        return imgYIQ

    #YIQ conversion matrix
    YIQ_mat = np.array([[ 0.299, 0.587, 0.114],
                        [ 0.596, -0.275, -0.321],
                        [ 0.212, -0.523, 0.311]])
    #RGB conversion matrix is the inverse of the YIQ conversion matrix
    RGB_mat = np.linalg.inv(YIQ_mat)

    #turn our NxMx3 matrix to (N*M)x3 matrix
    img_vals = imgYIQ.reshape((imgYIQ.shape[0] * imgYIQ.shape[1], 3))

    #matrix multiplication to convert each YIQ triplet into RGB
    img_vals = np.matmul(img_vals,RGB_mat)

    #reshape our matrix back
    return img_vals.reshape(imgYIQ.shape).astype(float)

    


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img = imgOrig
    #handle case for inputing RGB
    if len(imgOrig.shape) == 3:
        try:
            ensure3ChannelImgInput(imgOrig)
        except Exception as e:
            print(e)
            return
        #get only the Y channel of our RGB image converted to YIQ
        img = transformRGB2YIQ(img)
        imEq, histOrg, histEq = handle_hist(img[:,:,0])
        img[:,:,0] = imEq
        img = transformYIQ2RGB(img)
        return img,histOrg,histEq
    #handle case for inputing image that isn't grayscale or RGB
    elif len(imgOrig.shape) != 2:
        print("Image must be RGB or grayscale!")
        return
    
    return handle_hist(img)

    


def handle_hist(img)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #will do histogram equalization on some color channel 
    histOrg,bins = np.histogram(img*255,bins=256,range=[0,255])
    cumSum = np.cumsum(histOrg)
    imEq = img*255   
    
    for i in range(256):
        lut =  np.ceil(255.0*float(cumSum[i])/float(cumSum[-1]))
        imEq[np.logical_and(img*255>=bins[i],img*255<bins[i+1])] = lut
        #private case where the last bin is for values in range [254,255] instead of [254,255)
        if i==255:
            imEq[np.logical_and(img*255>=bins[i],img*255==bins[i+1])] = lut
    #new histogram
    histEq,_ = np.histogram(imEq,bins=256,range=[0,255])
    imEq = imEq/255

    return imEq, histOrg, histEq


    



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if nQuant > 255:
        return [],[]
    img = imOrig
    #handle case for inputing RGB
    if len(imOrig.shape) == 3:
        try:
            ensure3ChannelImgInput(imOrig)
        except Exception as e:
            print(e)
            return [],[]
        #get only the Y channel of our RGB image converted to YIQ
        img = transformRGB2YIQ(img)
        qImage_i_Y, error_i = handle_quntize(img[:,:,0],nQuant,nIter)
        qImage_i = []
        #for every new Y channel add it to qImage_i together with the original I and Q channels
        for y in qImage_i_Y:
            tempImg = np.copy(img)
            tempImg[:,:,0] = y
            tempImg = transformYIQ2RGB(tempImg)
            qImage_i.append(tempImg)
        return qImage_i, error_i
    #handle case for inputing image that isn't grayscale or RGB
    elif len(imOrig.shape) != 2:
        print("Image must be RGB or grayscale!")
        return [],[]
    
    return handle_quntize(img,nQuant,nIter)
    
def handle_quntize(img: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    #if an image is using less colors than nQuant then we just apply our quantization on all colors (should return the original img)
    if len(np.unique(img))<=nQuant:
        return [img for i in range(nIter)], [0 for i in range(nIter)]

    hist,_ = np.histogram(img*255,bins=256,range=[0,255])
    z = [0]
    total_px = sum(hist)
    i = 0
    #initial spread of cells
    while i < len(hist):
        pixel_count = 0
        while pixel_count <= total_px/(nQuant) and i < len(hist):
            pixel_count+=hist[i]
            i+=1
        z.append(i)
    #set the last cell to 255 instead of 256
    z[-1]=255
    images = []
    errors = []
    #process each iteration
    for i in range(nIter):
        newImg,err,q = quantize_iter(img,hist,z)
        z = [0]
        for j in range(1,len(q)):
            z.append(round((q[j]+q[j-1])/2))
        z.append(255)
        images.append(newImg)
        errors.append(err)
        
    return images,errors


def quantize_iter(img,hist,z) -> Tuple[np.ndarray,float,list]:
    #handle each iteration
    q = []
    newImg = np.copy(img)
    for k in range(len(z)-1):
        #if some section is empty -> hist[z[k]:z[k+1]] is empty then no need to change that q
        if sum(hist[z[k]:z[k+1]]) == 0:
            q_i = round((z[k]+z[k+1])/2)
        else:
            #set q_i as the weighted average in every cell
            q_i = np.average([i for i in range(z[k],z[k+1])],weights=hist[z[k]:z[k+1]])
        
        newImg[np.logical_and(img*255>=z[k],img*255<z[k+1])] = q_i/255
        q.append(q_i)
    #calc MSE, note that we need to use sum twice since we are using a matrix
    error = np.sqrt(sum(sum(np.square(img-newImg))))/sum(hist)
    
    return newImg,error,q
    
    

def ensure3ChannelImgInput(img:np.ndarray):
    if len(img.shape) !=3 or img.shape[2] != 3:
        raise ValueError("Given image doesn't have 3 channels")



