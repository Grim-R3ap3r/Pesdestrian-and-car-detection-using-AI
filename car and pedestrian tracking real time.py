#for a stationary img ucan read the image manually
import cv2

# load some pre trained data and classify
car_tracker_file='car_detector.xml'

pedestrian_tracker_file='haarcascade_fullbody.xml'

car_tracker=cv2.CascadeClassifier(car_tracker_file)

pedestrian_tracker=cv2.CascadeClassifier(pedestrian_tracker_file)
print(type(pedestrian_tracker))




#to capture video from video(for frames)
video=cv2.VideoCapture('traffic.mp4') #0 stands for default webcam...for detecting faces in a video write name.mp4 instead of 0

#iterate forever over frames
while True:

    #read the current fame
    (read_successful,frame)=video.read()  #we only want the frame


    #converting a colour image to greyscale s that we dont have to deal with lot of colours
    grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    #detect cars and pedestrians
    car_coordinates=car_tracker.detectMultiScale(grayscaled_frame)  

    #detect pedestrians 
    pedestrian_coordinates=pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars(a&y coordinates,colour,thickness of the rectangle)
    #w&h are width and height of the rectangle
    for (x,y,w,h) in car_coordinates:
         cv2.rectangle(frame , (x,y), (x+w,y+h), (0, 0, 255), 2) 
    
    for (x,y,w,h) in pedestrian_coordinates:
        cv2.rectangle(frame , (x,y), (x+w,y+h), (0, 255, 255), 2) 

 
    #poping up the image to be detected
    cv2.imshow('AI based Car AND Pedestrian detection by Rathin', frame)
    key=cv2.waitKey(1)

    #stop if Q key is pressed
    if key==81 or key==113:
         break



