from keras.models import Sequential
from keras.layers import Dense

from numpy import array
from keras.models import load_model
import numpy as np
import cv2

params = cv2.SimpleBlobDetector_Params()



# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.01

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = .01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)
# GET ROI

# read image
img = cv2.imread('ROI_002.png')
# get RGB Values

average_color_per_row = np.average(img, axis=0)
average_color = np.average(average_color_per_row, axis=0)
average_color = np.uint8(average_color)

# Crack Detection
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(bw, 100, 200)
# cv2.imshow("Edges", bw)

average_crack_per_row = np.average(edges, axis=0)
average_crack = np.average(average_crack_per_row, axis=0)
average_crack = np.uint8(average_crack)
np_arr2 = np.array(average_crack)

keypoints = detector.detect(img)
nblobs = len(keypoints)
np_arr3 = np.array(nblobs)

# ave = np.concatenate(average_color, np_arr2)
ave = np.hstack((average_color, np_arr2, np_arr3)).ravel()

print(ave)

# generate 2d classification dataset

# define and fit the final model

# load model
model = load_model('RGB_CRACK_BLOBS_V2.h5')
# summarize model.
model.summary()


'''model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=500, verbose=0)'''
# new instance where we do not know the answer
Xnew = array([ave])
# make a prediction
ynew = model.predict_classes(Xnew)
# show the inputs and predicted outputs
print("X=%s Predicted=%s" % (Xnew[0], ynew[0]))
if ynew[0] == [0]:
    print('Reject')
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
    #cv2.imshow('Result', img)
else:
    print('Good')
