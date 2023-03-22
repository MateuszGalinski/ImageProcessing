from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tqdm import tqdm

mask_size = 71

def singleTreshholding(pixelsArray, xsize, ysize):
    imageArray = pixelsArray.copy()
    for j in range(xsize):
        for i in range(ysize):
            if(int((np.sum(pixelsArray[i][j])/3))>125):
                imageArray[i][j]=0
            else:
                imageArray[i][j]=255

    newImageYoda = Image.fromarray(imageArray)
    #plt.imsave("singleThreshholding.jpeg", newImageYoda)
    newImageYoda.save("singleThreshholding.jpeg")
    #newImageYoda.show()

def doubleTreshholding(pixelsArray, xsize, ysize):
    imageArray = pixelsArray.copy()
    for j in range(xsize):
        for i in range(ysize):
            if(int((np.sum(pixelsArray[i][j])/3))>150 or int((np.sum(pixelsArray[i][j])/3))<50):
                imageArray[i][j]=0
            else:
                imageArray[i][j]=255

    newImageYoda = Image.fromarray(imageArray)
    #plt.imsave("doubleThreshholding.jpeg", newImageYoda)
    newImageYoda.save("doubleThreshholding.jpeg")
    #newImageYoda.show()

def mapColours(pixelsArray, xsize, ysize, imgShape):

    for j in range(xsize):
        for i in range(ysize):
            pixelsArray[i][j]=np.sum(pixelsArray[i][j]/3)

    imageArray2 = pixelsArray.astype(np.uint8)
    plt.imsave("imageArrayNotNormalized.jpeg", imageArray2)

    img = pixelsArray
    ogPixelsArray = pixelsArray.flatten()

    unique, counts = np.unique(pixelsArray, return_counts=True)

    percent = counts / np.sum(counts)

    grayMap = np.asarray((unique,counts)).T

    cdf = [0 for i in range(256)]
    hv = [0 for i in range(256)]

    for i in range(256):
        cdf[i]+=float((cdf[i-1]+percent[i]))
        hv[i]=int(255*cdf[i])
    
    imageArray = [hv[i] for i in ogPixelsArray]
    imageArray = np.reshape(np.asarray(imageArray), imgShape)
    imageArray = imageArray.astype(np.uint8)
    plt.imsave("imageArrayNormalized.jpeg", imageArray)

def countPixels(pixelsArray, i, j):
    s = np.sum(pixelsArray[i-mask_size//2:i+mask_size//2+1, j-mask_size//2:j+mask_size//2+1, 0])/(mask_size**2)
    # print(s)

    return s

def filterAplication(pixelsArray, xsize, ysize):
    print("To grayscale")
    for j in tqdm(range(xsize)):
        for i in range(ysize):
            pixelsArray[i][j]=np.sum(pixelsArray[i][j]/3)

    print("Applying filter")
    imageArray2 = pixelsArray.copy()
    for j in tqdm(range(mask_size//2,xsize-mask_size//2)):
        for i in range(mask_size//2,ysize-mask_size//2):
            imageArray2[i][j]=countPixels(pixelsArray, i, j)

    blurredImage = Image.fromarray(imageArray2)
    plt.imsave("imageBlurred.jpg", imageArray2)

def tableMethod(pixelsArray, xsize, ysize):
    print("To grayscale")
    for j in tqdm(range(xsize)):
        for i in range(ysize):
            pixelsArray[i][j]=np.sum(pixelsArray[i][j]/3)
    print("Creating table\n\n")
    imageArray2 = pixelsArray.copy()
    tableOfValues = pixelsArray.copy()
    tableOfValues = (tableOfValues.cumsum(axis = 0).cumsum(axis = 1))
    print("\n\nAppplying filter")
    for j in tqdm(range(mask_size//2, xsize-mask_size//2)):
        for i in range(mask_size//2, ysize-mask_size//2):
            A = int(tableOfValues[i - mask_size//2, j + mask_size//2, 0])
            B = int(tableOfValues[i + mask_size//2, j + mask_size//2, 0])
            C = int(tableOfValues[i - mask_size//2, j - mask_size//2, 0])
            D = int(tableOfValues[i + mask_size//2, j - mask_size//2, 0])
            #print(A + D - C - B)
            imageArray2[i][j] = (-1)*(A + D - C - B)/mask_size**2
    blurredImage = Image.fromarray(imageArray2)
    plt.imsave("imageBlurredTableMethod.jpg", imageArray2)
    blurredImage.show()

def main():
    ##################################################################################
    imageYoda = Image.open("yoda.jpeg", mode='r', formats=None)
    xsize, ysize = imageYoda.size
    pixelsYoda = np.array(imageYoda)
    ##################################################################################
    imageRoad = Image.open("road.jpg", mode='r', formats=None)
    xsizeR, ysizeR = imageRoad.size
    pixelsRoad = np.array(imageRoad)
    ##################################################################################
    #imageRoad = Image.open("road.jpg", mode='r', formats=None)
    #xsizeR, ysizeR = imageRoad.size
    #pixelsRoad = np.array(imageRoad)
    ##################################################################################
    #print(pixelsYoda.shape)
    #singleTreshholding(pixelsYoda, xsize, ysize)
    #doubleTreshholding(pixelsYoda, xsize, ysize)
    mapColours(pixelsYoda, xsize, ysize, pixelsYoda.shape)
    filterAplication(pixelsRoad, xsizeR, ysizeR)
    tableMethod(pixelsRoad, xsizeR, ysizeR)

if __name__ == "__main__":
    main()    