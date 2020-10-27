import numpy as np
import cv2
import os

SHAPES = [1, 2, 3, 4]
CROSS = 3
SHAPE_NO_CROSS = SHAPES.copy()
SHAPE_NO_CROSS.remove(CROSS)
COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_DICT = {
    "0": (200, 0, 0),  # blue
    "1": (0, 200, 0),  # green
    "2": (0, 0, 200),  # red
    "3": (100, 0, 100),  # purple
    "4": (0, 250, 250),  # yellow
    "5": (125, 0, 250),  # pink
    "6": (0, 0, 0),  # black
    "7": (0, 125, 250),  # orange
    "8": (50, 50, 125),  # brown
    "9": (125, 125, 125),  # gray
}
LINE_THICKNESS = 1
THICKNESS = -1
HEIGHT = 10
WIDTH = 12
SIZE = 10
HOME = "/home-net/home-2/jtenegg1@jhu.edu"


def numberOfShapes(density=10, nmax=HEIGHT * WIDTH):

    n = np.ceil(np.random.exponential(density))
    n = min(n, nmax)
    return n


def addElement(im, x, y, shape, color):

    im[x, y, 0] = shape
    im[x, y, 1] = color
    return im


def createRandomImageWithFixedCrosses(n, crosses, h=HEIGHT, w=WIDTH):

    im = np.zeros((h, w, 2), dtype=int)
    count = 0
    count_crosses = 0

    for i in range(h):

        for j in range(w):

            if count_crosses < crosses or count < n:
                # ADD CROSSES TO REACH DESIRED NUMBER OF CROSSES
                if count_crosses < crosses:
                    im = addElement(im, i, j, 3, np.random.choice(COLORS))
                    count_crosses += 1
                    count += 1
                elif count_crosses == crosses:
                    # ADD OTHER SHAPES (MAKE SURE THERE ARE NO OTHER CROSSES)
                    im = addElement(
                        im,
                        i,
                        j,
                        np.random.choice(SHAPE_NO_CROSS),
                        np.random.choice(COLORS),
                    )
                    count += 1

    for i in range(h):
        np.random.shuffle(im[i, :, :])  # shuffle each row
    for j in range(w):
        np.random.shuffle(im[:, j, :])  # shuffle each column

    return im


def saveImage(im, crosses, number):

    image = 255 * np.ones((HEIGHT * SIZE, WIDTH * SIZE, 3), dtype=np.uint8)

    for i in range(HEIGHT):

        for j in range(WIDTH):

            shape = im[i, j, 0]
            # THERE IS A SHAPE
            if shape != 0:
                color = COLOR_DICT[str(im[i, j, 1])]

            # SHAPE IS A CIRCLE
            if shape == 1:
                center = (int(SIZE * (j + 0.5)), int(SIZE * (i + 0.5)))
                radius = int(0.9 * SIZE / 2)
                image = cv2.circle(image, center, radius, color, THICKNESS)

            # SHAPE IS A SQUARE
            elif shape == 2:
                top_left = (int(SIZE * (j + 0.1)), int(SIZE * (i + 0.1)))
                bottom_right = (
                    int(top_left[0] + SIZE * 0.8),
                    int(top_left[1] + SIZE * 0.8),
                )
                image = cv2.rectangle(image, top_left, bottom_right, color, THICKNESS)

            # SHAPE IS A CROSS
            elif shape == 3:
                top_left = (int(SIZE * (j + 0.1)), int(SIZE * (i + 0.1)))
                top_right = (int(top_left[0] + SIZE * 0.8), int(top_left[1]))
                bottom_left = (int(top_left[0]), int(top_left[1] + SIZE * 0.8))
                bottom_right = (
                    int(top_left[0] + SIZE * 0.8),
                    int(top_left[1] + SIZE * 0.8),
                )
                image = cv2.line(image, top_left, bottom_right, color, LINE_THICKNESS)
                image = cv2.line(image, top_right, bottom_left, color, LINE_THICKNESS)

            # SHAPE IS A TRIANGLE
            elif shape == 4:
                top_left = (int(SIZE * (j + 0.1)), int(SIZE * (i + 0.1)))
                bottom_left = (int(top_left[0]), int(top_left[1] + SIZE * 0.8))
                bottom_right = (
                    int(top_left[0] + SIZE * 0.8),
                    int(top_left[1] + SIZE * 0.8),
                )
                top = (int(0.5 * (bottom_left[0] + bottom_right[0])), top_left[1])
                triangle_cnt = np.array([bottom_left, bottom_right, top])
                cv2.drawContours(image, [triangle_cnt], 0, color, -1)

    filename = os.path.join(
        HOME, "data/Jacopo/HShap/LOR/%d/ex%d_%d.png" % (crosses, crosses, number)
    )
    cv2.imwrite(filename, image)
    print("Saved image %d_%d" % (crosses, number))


def main():

    # NUMBER OF CROSSES IN LOR DATASET
    crosses_classes = [1, 2, 3]
    # NUMBER OF IMAGES PER CROSS CLASS
    N = 100
    for crosses in crosses_classes:
        for i in range(N):
            n = numberOfShapes()
            im = createRandomImageWithFixedCrosses(n, crosses)
            saveImage(im, crosses, i)


if __name__ == "__main__":

    main()
