
from ast import arg
from operator import mod
from time import time

from threading import Thread, Event
import keras as ks
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import time
import sys
import socketio
import copy
import random
import string

event = Event()

# we put some variables in the global scope so we can share the memory between threads
mask = None

steering = 90

steering_offset = 0
speed = 0

subLoopExecution = 1

sio = socketio.Client()

import argparse

# Parse the command line arguments below
# - neural network file name
# - car ip address
# - car speed

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--neural_network_file', type=str, default='model.h5', required=False, help='Neural network file name')
    parser.add_argument('-i', '--ip_address', type=str, default="192.168.0.10", required=False, help='Car ip address')
    parser.add_argument('-s', '--speed', type=int, default=50, required=False, help='Car speed value from 0 to 100')
    return parser.parse_args()


def main():
    printBanner()

    # Parse the command line arguments
    args = parse_arguments()
    printConfig(args)

    model = ks.models.load_model(args.neural_network_file)

   
    # we try to connect to the PiCar
    try:
        sio.connect('http://192.168.0.10:3000')
    # if we fail we print out an error message and exit the program
    except:
        print("Failed to connect to PiCar Socket Error")
        print("Check that your laptop is connected to the PiCar network")
        exit()



    # Run the control loop
    controlLoop(args.ip_address, sio, model)
    print("Program Ended")
    sys.exit()


def captureVideo(cap, ip):
    fps = 0
    new_frame_time = 0
    prev_frame_time = 0

    # we create a while loop to get and display the video
    global subLoopExecution, mask
    while subLoopExecution:
        
        # we get the next frame from the video stored in frame
        # ret is a boolean that tells us if the frame was successfully retrieved
        # ret is short for return
        ret, frame = cap.read()
        # we display the frame but first lets pass it back by reference
        # so we can use it in the main thread
        try:
            mask = imageProcessing(frame)
            cv.imshow("Mask", mask)
            fpsCounter(frame, fps)
            cv.startWindowThread()
            cv.imshow("PiCar Video", frame)
            #getControls()
        except:
            print("Error displaying frame")
            print("Have you connected to the PiCar?")
            print("Is %s the correct ip address?" % ip)
            exit()
        # we use the waitKey function to wait for a key press
        # if the key is q then we break out of the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            subLoopExecution = 0
            exit()
            break

        new_frame_time = time.time()
        #we calculate the fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    print("Main Thread Ended")

def controlLoop(ip, sio, model):
    
    cap = cv.VideoCapture("http://%s:8080/?action=stream" % ip)
    
    print("----------------------------- Car Information ----------------------------")
    print("Camera FPS:", cap.get(cv.CAP_PROP_FPS))
    print("Camera width:", cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print("Camera height:", cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print("--------------------------------------------------------------------------")
    print("Capturing video from {}".format(ip))
    print("Press 'q' to quit")
    print("----------------------------- Sending Commands ---------------------------")



    print("Initializing threads")
    t1 = Thread(target=sendCommands, args=(sio,), daemon=True)
    t2 = Thread(target=predictSteering, args=(model,), daemon=True)
    print("Starting threads")
    t1.start()
    t2.start()

    # we start these threads after the sub threads 
    # this is because the main thread blocks execution of the sub threads
    # so we need the main thread active during the initialization of the sub threads
    print("Main Thread")
    controls()
    captureVideo(cap, ip)
    exit()
    
    

    

def lerp(a, b, t):
    return a + (b - a) * t



def imageProcessing(frame):
    # we get the trackbar values
    hl = cv.getTrackbarPos('Hue Lower', 'Controls')
    sl = cv.getTrackbarPos('Sat Lower', 'Controls')
    vl = cv.getTrackbarPos('Val Lower', 'Controls')
    
    hu = cv.getTrackbarPos('Hue Upper', 'Controls')
    su = cv.getTrackbarPos('Sat Upper', 'Controls')
    vu = cv.getTrackbarPos('Val Upper', 'Controls')
    # we convert the frame to the HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # blur the image to remove noise
    blur = cv.GaussianBlur(hsv, (5, 5), 0)
    ## mask the image to get only the desired colors
    mask = cv.inRange(blur, (hl, sl, vl), (hu, su, vu))
    ## we erode and dilate to remove noise
    erode = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    dilate = cv.dilate(erode, np.ones((5, 5), np.uint8), iterations=1)
    # we smooth the image with some gaussian blur
    blur = cv.GaussianBlur(dilate, (5, 5), 0)

    return blur


def fpsCounter(image ,fps):
    cv.putText(image, "FPS:" + str(round(fps, ndigits=2)), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255))

def null(x):
    pass

def updateSpeed(x):
    global speed
    speed = x

def controls():
    # create cv2 window
    cv.namedWindow('Controls', cv.WINDOW_NORMAL)
    cv.resizeWindow('Controls', 300, 300)
    cv.createTrackbar('Hue Lower', 'Controls', 40, 255, null)
    cv.createTrackbar('Sat Lower', 'Controls', 25, 255, null) 
    cv.createTrackbar('Val Lower', 'Controls', 73, 255, null)

    cv.createTrackbar('Hue Upper', 'Controls', 93, 255, null)
    cv.createTrackbar('Sat Upper', 'Controls', 194, 255, null)
    cv.createTrackbar('Val Upper', 'Controls', 245, 255, null)

    cv.createTrackbar('Speed', 'Controls', 0, 100, updateSpeed)
    cv.createTrackbar('Steering Offset', 'Controls', 90 , 180, null)


def sendCommands(sio):
    print("Command Thread Started")
    while subLoopExecution:
        sio.emit('drive', speed)
        sio.emit('steer', steering)
        sys.stdout.write("\rspeed: %s steering angle: %s  " % (speed, steering))
        sys.stdout.flush()
    print("Command Thread Ended")
    return
    


def predictSteering(model):
    global mask, steering
    while subLoopExecution:

        maskref = copy.deepcopy(mask)

            # check dsize of mask
        try:
            maskref = cv.resize(maskref, (100, 66))
            maskref = np.array(maskref)
            maskref = maskref.reshape(1, 100, 66, 1)
            steering = float(model(maskref)[0][0])
        except:
            continue

    

def printConfig(args):
    print("--------------------------------- Config ---------------------------------")
    print("Neural network file name: {}".format(args.neural_network_file))
    print("Car ip address: {}".format(args.ip_address))
    print("Car speed: {}".format(args.speed))

def printBanner():
    # print raw string
    print(r"""
       ____ ___ ____             ____                              
      |  _ \_ _/ ___|__ _ _ __  |  _ \ _   _ _ __  _ __   ___ _ __ 
      | |_) | | |   / _` | '__| | |_) | | | | '_ \| '_ \ / _ \ '__|
      |  __/| | |__| (_| | |    |  _ <| |_| | | | | | | |  __/ |   
      |_|  |___\____\__,_|_|    |_| \_\\__,_|_| |_|_| |_|\___|_|   
      ______________________________________________________________                                                         
""")

if __name__ == '__main__':
    main()
  