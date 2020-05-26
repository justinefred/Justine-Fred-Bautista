from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('NEW_RGBwithCrack3.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:5] # number of columns in dataset except the last column for label
y = dataset[:,5] # number of columns in dataset except the last column for label
# define the keras model
model = Sequential()
model.add(Dense(240, input_dim=5, init='uniform', activation='relu')) #input_dim = number of columns that has numerical values
model.add(Dense(160, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#model.add(Dense(12, input_dim=3, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=1500, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
model.save("RGB_CRACK_BLOBS_V22.h5")
print("Saved model to disk")