'''
@Author: Yazeed Alzughaibi - 1847186
@Author: Mohammed Almaafi  - 1845137

The original code was taken from 
https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook

'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from os import environ

'''
Labels of dataset are as follows: 
from 0 - 9 English digits
from 10 - 19 Arabic digits
from 20 - 45 English letters
'''

'''
This part was added to give the user the option to train or test the model
'''
signal=True
print('Please choose:\n1-To train the model\n2-To test pre-learned model')
option = int(input())
while(signal):
    if(option==1):
        signal = False
    elif(option==2):
        signal = False
    else:
        print('Please enter a valid number:')

################################# Getting Data Ready #############################
data = pd.read_csv('DigitReco\\Project\\FullDataExtended.csv')
testData= pd.read_csv('DigitReco\\Project\\ArabicDigitTest.csv')
testData = np.array(testData)
data = np.array(data)
# the number of node is dynamic
num_of_node = np.amax(data, axis=0)[0]+1 # this line is used for determining the number of nodes in each layer
m , n = data.shape
np.random.shuffle(data)

num_of_test_data = 38000 # to split test and train data after shuffle

def splitData(data, num_of_test_data):
    test = data[0:num_of_test_data].T
    train = data[num_of_test_data:m].T
    return test, train

def splitXY(data):
    y = data[0]
    x = data[1:n]
    x = x/255.
    return y,x

test,train=splitData(data, num_of_test_data)
y_test , x_test = splitXY(test)
# if you want to test a part of the trained dataset comment the next two lines
np.random.shuffle(testData)
y_test , x_test = splitXY(testData.T)
y_train , x_train = splitXY(train)


################################# Applying NN #############################
'''
activation method
'''
def ReLU(z):
    return np.maximum(z,0)

def ReLU_deriv(z):
    return z > 0

def softmax(z):
    return (np.exp(z)/sum(np.exp(z)))


'''
initilazing parameters
'''
def init_params(num):
    # 45 (number of digits and letters) is the number of nodes in the hidden layers
    # it has input layer, two hidden layers, then output layer (yhat)
    w1 = np.random.rand(num,(n-1))-0.5# rand values between 0 and 1
    b1 = np.random.rand(num,1)-0.5
    w2 = np.random.rand(num,num)-0.5
    b2 = np.random.rand(num,1)-0.5 
    return w1,b1,w2,b2

# if you want to start from last check point 
def init_params1():
    w1 = pd.read_csv('W1.csv')
    w1 = np.array(w1)
    #b1
    b1 = pd.read_csv('b1.csv')
    b1 = np.array(b1)
    #w2
    w2 = pd.read_csv('W2.csv')
    w2 = np.array(w2)
    #b2
    b2 = pd.read_csv('b2.csv')
    b2 = np.array(b2)
    return w1,b1,w2,b2


'''
forward propagation
'''
def forward(w1,b1,w2,b2,x):
    # TODO use ReLU activation method for first layer, but last layer should use softmax because we want probabilty
    z1 = w1.dot(x)+b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1,a1,z2,a2

'''
one hot assignment 
'''
def one_hot(y):
    one_hot_y = np.zeros((y.size,y.max()+1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

'''
backward propagation
'''
def backward(z1,a1,a2,w2,x,y):
    # the same as forward, but backward.
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 =  w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2

'''
updating parameters
'''
def update(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - alpha * dw1 
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2 
    b2 = b2 - alpha * db2
    return w1,b1,w2,b2

'''
accuracy
'''
def accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

'''
gradient desceent
'''
if(option==1):
    def grad_desc(x,y,alpha, iter,num):
        signal2=True
        print('Please choose:\n1-for a new training\n2-for starting from previous training')
        option2 = int(input())
        while(signal2):
            if(option2==1):
                signal2 = False
            elif(option2==2):
                signal2 = False
            else:
                print('Please enter a valid number:')
        if(option2==1):
            w1, b1, w2, b2 = init_params(num)
        elif(option2==2):
            w1, b1, w2, b2 = init_params1()
        for i in range(iter):
            z1, a1, z2, a2 = forward(w1,b1,w2,b2,x)
            dw1, db1, dw2, db2 = backward(z1,a1,a2,w2,x,y)
            w1, b1, w2, b2 = update(w1,b1,w2,b2,dw1, db1, dw2, db2, alpha)
            if(i%10==0):
                print("Iteration %d out of %d "%(i,iter))
                prediction = np.argmax(a2,0)
                print('The accuracy is %.3f'%((accuracy(prediction,y)*100)))
                print('='*30)
        return w1, b1, w2, b2
    print('Please entre the number of iterations: ')
    w1, b1, w2, b2 = grad_desc(x_train, y_train, 0.1, int(input()), num_of_node)
    # to save w1, b1, w2, b2 values for pre-learned section
    with open("w1.csv",'w',   newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['test' for i in range(len(w1[0]))])
        for i in range(len(w1)):
            wr.writerow(w1[i])
        
    with open("b1.csv",'w',   newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['test' for i in range(len(b1[0]))])
        for i in range(len(b1)):
            wr.writerow(b1[i])
            
    with open("w2.csv",'w',   newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['test' for i in range(len(w2[0]))])
        for i in range(len(w2)):
            wr.writerow(w2[i])
            
    with open("b2.csv",'w',   newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(['test' for i in range(len(b2[0]))])
        for i in range(len(b2)):
            wr.writerow(b2[i])

'''
predicting
'''
def predicting(x, w1, b1, w2, b2):
    z1,a1,z2, a2 = forward(w1,b1,w2,b2,x)
    predictions = np.argmax(a2,0)
    return predictions

'''
testing
'''
def testing(index, w1, b1, w2, b2):
    curr_img = x_test[:, index, None]
    prediction = predicting(curr_img, w1, b1, w2, b2)
    label = y_test[index]
    # 1 for pred 2 for label
    flag1 = False
    flag2 = False
    flag3 = False
    tag1 = 'Arabic'
    tag2 = 'Arabic'
    tag3 = 'True prediction'
    letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    print('-'*60)
    if(prediction == label):
        flag3 = True
    tag3 = tag3 if flag3 else 'False prediction'
    if(prediction>9 and prediction<20):
        prediction=prediction-10
        flag1 = True
    tag1 = tag1 if flag1 else 'English'
    if(prediction>=20):
        prediction = prediction - 20
        prediction = letters[prediction[0]]
        tag1=''
    if(label>9 and label<20):
        label=label-10
        flag2 = True
    tag2 = tag2 if flag2 else 'English'
    if(label>=20):
        label = label - 20
        label = letters[label]
        tag2=''

    print(f'The machine has predicted ({tag2} {label} ) as an ({tag1} {prediction[0]} )  ')
    # print('The machine has predicted (%-7s  %d) as an (%-7s  %d)  |'%(tag1 ,prediction[0],tag2,label))
    print('%s'%(tag3))
    # environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    # environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    # environ["QT_SCALE_FACTOR"] = "1"
    # curr_img = curr_img.reshape((28,28))*255
    # plt.gray()
    # plt.imshow(curr_img, interpolation='nearest')
    # plt.show()


'''
pre-learned data 
'''
if(option==2):
    #w1 
    w1 = pd.read_csv('W1.csv')
    w1 = np.array(w1)
    #b1
    b1 = pd.read_csv('b1.csv')
    b1 = np.array(b1)
    #w2
    w2 = pd.read_csv('W2.csv')
    w2 = np.array(w2)
    #b2
    b2 = pd.read_csv('b2.csv')
    b2 = np.array(b2)
'''
Implementation
'''
test_pred= predicting(x_test,w1, b1, w2, b2)
accuracy(test_pred, y_test)
print('------------------------Test Exampls------------------------')
for i in range(7):
    testing(i, w1, b1, w2, b2)
print(accuracy(test_pred, y_test)*100)

