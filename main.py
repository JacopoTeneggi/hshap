################################################# HII BANAAS

import numpy as np # for vector manipulations
import graphics
import matplotlib.pyplot as plt # to experiment with histograms
import random

height = 8 # number of squares in the vertical direction of the image
width = 12 # " horizontal  "
N = height*width # number of squares
size = 70 # size of a square

circleColor = graphics.color_rgb(10,200,230)

colorDict = {
    "0": "blue",
    "1": "pink",
    "2": "red",
    "3": "black",
    "4": "red",
    "5": "orange",
    "6": "purple",
    "7": "green",
    "8": "yellow",
    "9": "brown"
}
def main():
    images = createData1(1,3,height,width) # contains information about the shapes and colors in the image
    # first coordingate: 0 is nothing, 1 is a circle
    # second coordinate: color (see above)


    for i in range(len(images)):
        win = graphics.GraphWin("Image", size * width, size * height)  # Graphics inverts x and y
        win.setBackground('white')
        drawImage(win, images[i])
        win.save("im.png")
        win.getMouse()
        win.close()


    saveData(images, "data1.npy")

    #print(loadData("data1.npy"))



def drawImage(w,im):

    for i in range(height):
        for j in range(width):
            if im[i,j, 0]:
                coord = (size*(j+0.5), size*(i+0.5)) # Graphics inverts x and y
                point = graphics.Point(coord[0], coord[1])
                circle = graphics.Circle(point, size/2)
                circle.setFill(colorDict[str(im[i,j,1])])
                circle.draw(w)

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

        for j in range(w):
            np.random.shuffle(im[:,j,:]) # shuffle each column

        np.random.shuffle(im) # shuffle the columns
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