# importing various modules
import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
import time


# images folder path from current directory
path='ImageAttendence'
# contain matrix of each and every image 
# stores every image after reading it
images=[]
# contains name of each image file after removing extention part
classNames=[]
# is used to get the list of all the files and directories in specified directory
myList=os.listdir(path) 
print(myList)


for cl in myList:
    # reads an image and return 0 matrix if not loaded from disk 
    # returns 2D matrix for a binary or grey scale image
    # returns 3D matrix for colored image
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # used to split the path name into pair root and ext
    # here ext is extention portion of specific path
    # root is everything except ext part
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList=[]
    for img in images:
        # converting BGR image to RGB
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Get 128-Dimension face encoding
        #Always return the list of found faces, for this purpose we take first image only
        #(assuming only one face per image)
        #method generates a 128-d real-valued number feature vector per face
        encode=fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



def markAttendance(name):
    #'r+' means you can both read and write file
    with open('Attendence.csv','r+') as f:
        #readlines() returns a list containing each line in the file as a list item
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            # each line contains name,time both the values seperated by comma
            # gives you the list of two items name and date
            entry=line.split(',')
            # first element in entry list is name and second is date
            # here we are appending the names 
            nameList.append(entry[0])
        if name not in nameList:
            #using now() to get current time
            now=datetime.now()
            # converting datetime object containing current date time into different string format
            dtString=now.strftime('%H:%M:%S')
            # write name and time in file in csv format
            f.writelines(f'\n{name},{dtString}')

# Passing the images list( which contains every read image ) to findEncoding() function
encodeListKnown=findEncodings(images)
# this will return the video from the first webcam on your laptop
# you can use 1 if you want external device to capture video 
cap=cv2.VideoCapture(0)
# returns time in seconds (floating point) since epoch
# the epoch is January 1, 1970, 00:00:00 
startTime=time.time()
seconds=20

while True:
    # after 20 seconds we want this loop to break and stop video capturing
    if(time.time()-startTime>seconds):
        break
    #cap.read() returns is a boolean (True/False) and image content.
    # it stores the read image in img and also saves a boolean value to success
    #  recording whether we were able to read the image
    #  If you remove success, the img variable takes that boolean and image data as a tuple.
    #  There you got an error
    success,img=cap.read() 
    # the larger the image, the more the data, and 
    # therefore the longer it takes for algorithms to process the data
    #By decreasing the image size, we have fewer pixels to process 
    # (not to mention less noise to deal with), which leads to faster
    #  and more accurate image processing algorithms.

    # syntax of resize function in opencv
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    #dsize - here we are not going to define any pixel size
    # fx and fy are the scale factors along horizontal and vertical axis
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #Returns an array of bounding boxes of human faces in a image
    #A list of tuples of found face locations (top, right, bottom, left) order
    faceCurFrame=fr.face_locations(imgS)
    #Given an image, return the 128-dimension face encoding for each face in the image
    encodeCurFrame=fr.face_encodings(imgS,faceCurFrame)
    
    #we wnat them in the same loop so thats why we are using Zip
    # outputs the list of tuples containing encodeFace,faceloc
    #example:list_a = [1,2,3,4]
    #list_b = ['a', 'b', 'c', 'd']
    #print(list(zip(list_a, list_b)))
    # Returns: [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
    for encodeFace,faceloc in zip(encodeCurFrame,faceCurFrame):
        #A list of True/False values indicating
        # which known_face_encodings match the face encoding to check
        matches=fr.compare_faces(encodeListKnown,encodeFace)
        #compare them to a known face encoding and get a euclidean distance 
        #for each comparison face. The distance tells you how similar the faces are.
        #return: an array                  
        faceDis=fr.face_distance(encodeListKnown,encodeFace)
        # returns index of min value in faceDis list
        # lower the distance is more similar the face is 
        matchIndex=np.argmin(faceDis)
        
        #if value of matches at match index is true
        if matches[matchIndex]:
            # get the name at matchIndex and making it uppercase
            name=classNames[matchIndex].upper()
            # getting top, right, bottom, left values of faceloc in y1,x2,y2,x1 variables
            #faceloc is a tuple which is contained in a list
            y1,x2,y2,x1=faceloc

            #above we reduce the scaling factor by 0.25 ie 1/4 means that we reduce the image size by 4
            #now we are normalizing it because we want it according to orignal image
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4

            # Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
            #   x1,y1 ------
            #   |          |
            #   |          |
            #   |          |
            #   --------x2,y2
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #to draw another rectangle just below the face rectangle and fill colour to green 
            cv2.rectangle(img,(x1,y2-32),(x2,y2),(0,255,0),cv2.FILLED)
            # putText() method is used to draw a text string on any image
            #Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            #org: It is the coordinates of the bottom-left corner of the text string in the image
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #passing name to the function markAttendance
            markAttendance(name)
    # show each and every frame 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)