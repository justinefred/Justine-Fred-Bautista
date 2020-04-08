"NaiveBayesTraining.py" file is to check the accuracy percentage of the "NaiveDataset.csv" dataset 
that is being used. 

"NaiveBayesPredict.py" file is to use the dataset with real image. Its process is to get the 
numerical values of test images ex. 1.jpg
1. It will get the Blue, Green, Red, Cracks, Spots features on the certain image and store it in
ave array [B,G,R,Cracks,Spots]

2. Using Naive Bayes Classifier and the trained dataset from previous file, and it will read the CSV 
file and predict whether the tomato is good or bad in quality. 





Credits to Jason Brownlee PhD (https://machinelearningmastery.com/)