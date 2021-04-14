import cv2
#our image
img_file='car_image.jpg'

#our pre-traine car classifier
classifier_file ='car_detector.xml'

#ceate opencv image
img=cv2.imread(img_file)

#create classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

#convert  to grayscale(needed for haarcascade)
black_white=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars=car_tracker.detectMultiScale(black_white)

#draw rectangles around the cars
for(x,y,w,h) in cars:
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)

#display the image with the faces spotted
cv2.imshow('CAR AND PADESTRIANS TRACKER BY RATHIN',img)

#dont autoclose(wait here in the code and listen for a key press)
cv2.waitKey()