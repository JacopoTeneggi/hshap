import numpy as np # for vector manipulations
import cv2
import os
import matplotlib.pyplot as plt # to experiment with histograms
import random
import shutil


height = 10 # number of squares in the vertical direction of the image
width = 12 # " horizontal  "
N = height*width # number of squares
size = 70 # size of a square

colorDict = {
    "0": (200, 0, 0), # blue
    "1": (0,200,0), # green
    "2": (0,0,200), # red
    "3": (100,0,100), # purple
    "4": (0,250,250), # yellow
    "5": (125, 0, 250), # pink
    "6": (0,0,0), # black
    "7": (0,125,250), # orange
    "8": (50, 50, 125), # brown
    "9": (125,125,125) # gray
}
def main():
    n0 = 100
    n1 = 1000

    images = createData1(n0,n1,height,width) # contains information about the shapes and colors in the image
    # first coordingate: 0 is nothing, 1 is a circle
    # second coordinate: color (see above)


    for i in range(len(images)):
        if not os.path.exists("data1"):
            os.makedirs("data1/0")
            os.makedirs("data1/1")
        saveImage(i, images[i], i >= n0, False) # Careful: overwrites but doesn't delete previous data!



    saveData(images, "data1.npy")

    #print(loadData("data1.npy"))


def saveImage(number, im, label, draw):
    image = 255*np.ones((height*size, width*size, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if im[i,j,0]:
                coord = (int(size*(j+0.5)), int(size*(i+0.5))) # (x,y)
                color = colorDict[str(im[i,j,1])]
                thickness = -1
                image = cv2.circle(image, coord, int(size/2), color, thickness)

    filename = "data1/"

    if label:
        filename += "1/"
    else:
        filename += "0/"

    filename += "image" + str(number) + ".png"

    cv2.imwrite(filename, image)
    if draw:
        cv2.imshow('Window', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




def createImage1(label, h, w): # RULE: 1 if there is at least one circle
    im = np.zeros((h, w, 2), dtype=int) # im[0] is the shape, im[1] the color

    if label:
        n = np.int(np.random.exponential(5))
        n = min(n,N)
        n = max(n,1)
        k = 0
        cont = True
        for i in range(h):
            for j in range(w):
                if cont:
                    im[i,j,0] = 1
                    im[i,j,1] = np.random.randint(0,10)
                    k += 1
                    if k >= n:
                        cont = False

        for i in range(h):
            np.random.shuffle(im[i,:,:]) # shuffle each row
        for j in range(w):
            np.random.shuffle(im[:,j,:]) # shuffle each column

        #np.random.shuffle(im) # mix the rows


    return im

def createData1(n0, n1, h, w):
    IM = []
    ind = 0
    for i in range(n0):
        IM.append(createImage1(0,h,w))
        ind += 1
    for i in range(n1):
        IM.append(createImage1(1,h,w))
        ind += 1
    return IM

def saveData(ims, filename):
    np.save(filename, ims)

def loadData(filename):
    return np.load(filename)


main()