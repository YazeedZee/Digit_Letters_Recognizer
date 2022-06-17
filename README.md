# Digit_Letters_Recognizer
This project is supposed to take in a csv file of English numbers and letters as well as Arabic numbers (28x28 pixels) to train and test to be able to recognize them.

# Description
The code is a “from scratch” (meaning no machine learning libraries were used) neural network (NN) of four layers, two of them are hidden and the other two are input/output.
The original code was only applied to MNIST dataset of English numbers and every run needs to retrain the model, however we took the code and modified it substantially in such a way that it can process English numbers and letters as well as Arabic numbers. Also the code can process virtually any other type of data as long as it is in csv file format. 
Mainly, we changed the way the code receives data, it can test on pre-learned model using parameters from previous training parameters, it can be trained from scratch or continue from last training checkpoint at user specified number of iterations. User can train and test on separate runs/executions. The user can always improve the model accuracy whenever they need to. One of our experiments was adding more layers to the NN, when we tried to add an extra layer to the model using “leaky_ReLU” to avoid dying “ReLU”, the accuracy was very bad and wouldn’t get more than 5% training accuracy and we stayed with 2 layers since it worked well.
To train the model, we combined the data from MNIST (for English letters and numbers) and an Arabic data set from Kaggle (Arabic numbers) into one csv file and trained the model using it. Training might take a long time if you want a reasonable accuracy so we advise you to use the pre-learned code using the included w1, b1, w2 and b2 (based on a 50,000-iteration trained model that we trained)csv files and just test the code.
For testing however, first we tried to come up with our own test data, in which we write numbers and letter and take their pictures and essentially turn them into csv files using the “PIL” Python library, unfortunately our model requires any input data to be in MNIST style and cannot read others very well thus making the data that we made useless. Although for testing at the end we just took a partition of the full data (English numbers and letters as well as Arabic numbers) and used it for testing.

# Constraints
To use this code, there are some constraints:
-	Data must be in MNIST style
-	Pictures must be 28 x 28 pixels 
-	csv file must have the first column as labels for the data

# Results
The model was able to train up to an accuracy of 90% and testing accuracy was varying:
-	English letters testing got almost 80% accuracy.
![English letters](https://i.imgur.com/ktu6F9C.png)
-	English numbers testing got almost 95% accuracy.
![English numbers](https://i.imgur.com/KYmY05J.png) 
-	Arabic numbers testing got almost 96% accuracy. 
![Arabic numbers](https://i.imgur.com/nUp1Ac6.png)


# Reference
- Our main source for the code and English numbers dataset is this Kaggle page https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook 
- for the English letters dataset: https://www.kaggle.com/crawford/emnist 
- for the Arabic numbers dataset: https://www.kaggle.com/mloey1/ahdd1/metadata 
