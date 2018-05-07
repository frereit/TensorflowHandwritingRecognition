import cv2
import numpy as np
import predict as p


image = np.ones((640, 640, 1))


def click(event, x, y, flags, param):
    global image
    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(center=(x, y), radius=20, img=image, color=(0, 0, 0), thickness=-1)


def main():
    global image
    cv2.namedWindow("Input")
    cv2.setMouseCallback("Input", click)
    output = np.ones((512, 512, 1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (1, 511)
    fontScale = 23
    fontColor = (0, 0, 0)
    lineType = 2
    while True:
        cv2.imshow("Input", image)
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("f"):
            cv2.destroyAllWindows()
            break
        if key == ord("r"):
            image = np.ones((640, 640, 1))
        if key == ord("p"):
            clone = image.copy()
            clone = cv2.resize(clone, (32,32))
            final = np.zeros((32, 32, 1))
            for x in range(len(clone)):
                for y in range(len(clone[x])):
                    final[x][y][0] = clone[x][y]
            pred = p.predict(final)
            print("Predicted " , pred)
            output = np.ones((512, 512, 1))
            cv2.putText(output, pred, (10, 500), font, fontScale, fontColor, 10,  2)





if __name__ == "__main__":
    main()