import imutils
import cv2
import numpy as np
import tkinter as tk
from queue import Queue

width = 600
frames_count = 10

root = tk.Tk()
canvas1 = tk.Canvas(root, width=500, height=450)
canvas1.pack()
label1 = tk.Label(root, text='Video source:')
canvas1.create_window(250, 60, window=label1)
entry1 = tk.Entry(root, width=50)
canvas1.create_window(250, 100, window=entry1)
label2 = tk.Label(root, text='Sensitiveness threshold:')
canvas1.create_window(250, 140, window=label2)
entry2 = tk.Entry(root, width=50)
canvas1.create_window(250, 180, window=entry2)
mask_bool = tk.BooleanVar()
check2 = tk.Checkbutton(root, text="Mask", variable=mask_bool)
canvas1.create_window(250, 400, window=check2)
label3 = tk.Label(root, text='Mask upper left corners: (format: x1 y1 x2 y2 ...)')
canvas1.create_window(250, 220, window=label3)
entry3 = tk.Entry(root, width=50)
canvas1.create_window(250, 260, window=entry3)
label4 = tk.Label(root, text='Mask lower right corners: (format: x1 y1 x2 y2 ...)')
canvas1.create_window(250, 300, window=label4)
entry4 = tk.Entry(root, width=50)
canvas1.create_window(250, 340, window=entry4)
debug = tk.BooleanVar()
check1 = tk.Checkbutton(root, text="Debug", variable=debug)
canvas1.create_window(250, 380, window=check1)

last_frames = Queue()
def run():
    source = entry1.get()
    sensitiveness = int(entry2.get())
    mask_upper_left = []
    mask_lower_right = []
    if mask_bool.get():
        upper_left = entry3.get().split()
        lower_right = entry4.get().split()
        for i in range(0, len(upper_left), 2):
            mask_upper_left.append(tuple(map(int, upper_left[i:i+2])))
            mask_lower_right.append(tuple(map(int, lower_right[i:i+2])))

    vs = cv2.VideoCapture(source)
    last_frame = None
    while True:
        ret, frame = vs.read()

        if ret == False:
            break

        if mask_bool.get():
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            for i in range(len(mask_upper_left)):
                cv2.rectangle(mask, mask_upper_left[i], mask_lower_right[i], (255, 255, 255), -1)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            masked = frame

        frame = imutils.resize(frame, width=width)
        masked = imutils.resize(masked, width=width)

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if last_frame is None:
            last_frame = gray
            continue
        last_frames.put(gray)
        if last_frames.qsize() == frames_count:
            frameDelta = cv2.absdiff(last_frames.get(), gray)
        else:
            frameDelta = cv2.absdiff(last_frame, gray)
        thresh = cv2.threshold(frameDelta, sensitiveness, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)


        for c in cnts:
            if cv2.contourArea(c) < 50:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Motion Detector", frame)
        if debug.get():
            if mask_bool.get():
                cv2.imshow("Mask", masked)
            cv2.imshow("Grayscale", gray)
            cv2.imshow("Frame Delta", frameDelta)
            cv2.imshow("Thresh", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        last_frame = gray
    vs.release()
    cv2.destroyAllWindows()


button1 = tk.Button(text='Start', command=run)
canvas1.create_window(250, 430, window=button1)

root.mainloop()

