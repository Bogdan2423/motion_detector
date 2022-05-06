import imutils
import cv2
import numpy as np


vs = cv2.VideoCapture("./krakow.mp4")
sensitiveness = 20
debug = True
width = 900

# initialize the first frame in the video stream
first_frame = None
# loop over the frames of the video
while True:
    # grab the current frame
    ret, frame = vs.read()
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if ret == False:
        break

    #make mask
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (0, 0), (1000, 900), (255, 255, 255), -1)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    #resize original and masked frame
    frame = imutils.resize(frame, width=width)
    masked = imutils.resize(masked, width=width)

    # convert frame to grayscale, and blur it
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if first_frame is None:
        first_frame = gray
        continue
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frameDelta, sensitiveness, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the frame and record if the user presses a key
    cv2.imshow("Motion Detector", frame)
    if debug:
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        cv2.imshow("Mask", masked)
        cv2.imshow("Grayscale", gray)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()
