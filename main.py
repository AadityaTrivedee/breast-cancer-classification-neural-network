import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from termcolor import colored

#Extracting data
data = pd.read_csv('breast-cancer.csv', encoding='latin-1')
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

#Initialising parameters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_train, X_test, y_train, y_test = np.array(X_train, dtype = np.float64), np.array(X_test, dtype = np.float64), np.array(y_train, dtype = np.float64).reshape(-1,1), np.array(y_test, dtype = np.float64).reshape(-1,1)
X_train, X_test, y_train, y_test = X_train.astype("float64"), X_test.astype("float64") , y_train.astype("float64") , y_test.astype("float64")
dim = int(X_train.shape[1])
n_x,n_h,n_y = X_train.shape[1],int(2*dim/3),y_train.shape[1]
w1,b1 = np.random.randn(n_x, n_h)*0.01,np.zeros((1,n_h))
w2,b2 = np.random.randn(n_h, n_y)*0.01,np.zeros((1,n_y))
J, J_plt = float(0), []

print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape: ",y_test.shape)
print("w1 Shape: ",w1.shape)
print("w2 Shape: ",w2.shape)
print("b1 Shape: ",b1.shape)
print("b2 shape: ",b2.shape)


def scatter_plt():
    q1,q2 = 0,0
    for i in range(1,y_train.shape[0] +1):
        if y_train[i-1] == 0:
            q1 += 1
        elif y_train[i-1] == 1:
            q2 += 1
    for j in range(1,dim):
        X_b0, X_c0 = np.zeros((q1,1)), np.zeros((q1,1))
        X_b1, X_c1 = np.zeros((q2,1)), np.zeros((q2,1))
        t1,t2 = 0,0
        for i in range(1,y_train.shape[0] + 1):
            if y_train[i-1] == 0:
                t1 += 1
                X_b0[t1-1] += X_train[i-1,j-1]
                X_c0[t1-1] += X_train[i-1,j]
            elif  y_train[i-1] == 1:   
                t2 += 1
                X_b1[t2-1] += X_train[i-1,j-1]
                X_c1[t2-1] += X_train[i-1,j]
        
        plt.scatter(X_b0,X_c0,color= "r", marker = "o")
        plt.scatter(X_b1,X_c1,color="b", marker ="x")
    plt.grid()

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def hypo(x):
    Z1 = np.dot(x,w1)+ b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1,w2) + b2
    A2 = sigmoid(Z2)
    hypos = {"A2":A2,"Z2":Z2,"A1":A1}
    return hypos

def algo(alpha,):
    global w1,b1,w2,b2,J
    hypos = hypo(X_train)
    h,A1 = hypos["A2"],hypos["A1"]
    J = -(1/dim)*(np.dot(y_train.T,np.log(h))+np.dot((1-y_train.T),(np.log(1-h))))
    J_plt.append(J)
    dZ2 = h - y_train
    dW2 = (1/dim) * np.dot(A1.T,dZ2)
    db2 = (1/dim) * np.sum(dZ2, axis = 0, keepdims = True)
    dZ1 = (dZ2*w2.T)*(1 - np.power(A1, 2))
    dW1 = (1/dim) * np.dot(X_train.T,dZ1)
    db1 = (1/dim) * np.sum(dZ1, axis = 0, keepdims = True)
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dW2
    b2 -= alpha * db2

def cost_plt(itera):
    global J_plt
    x = np.array(np.linspace(1,itera,itera)).reshape(-1,1)
    J_plt = np.array(J_plt).reshape(-1,1)
    plt.plot(x,J_plt)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost function')
    plt.grid()
    plt.show()

def line_eqn():
    scatter_plt()
    temp = np.linspace(0,10,1000)
    temp = temp.reshape(1000,1)
    llx = np.tile(temp, (1, dim))
    hypos = hypo(llx)
    H = hypos["Z2"]
    plt.plot(llx,H,"g")
    plt.ylim(0,10)
    plt.grid()
    plt.show()

def accuracy(x,y):
    hypos = hypo(x) 
    h = hypos["A2"]
    y_pred = np.zeros((h.shape[0],h.shape[1]))
    for i in range(1,h.shape[0]):
        if h[i-1] > 0.5:
            y_pred[i-1] += 1
        elif h[i-1] <= 0.5:
            y_pred[i-1] += 0
    r1 =  1 - np.abs(np.sum((h - y))/np.sum(y))
    r2 =  1 - np.abs(np.sum((y_pred - y))/np.sum(y))
    res = {"Raw":r1, "Clean":r2 }
    return res

def predict():
    features = data.columns[:-1].tolist()
    given = []
    for i in range(1,dim+1):
        a =  float(input(f"\nEnter the required value for {features[i-1]} "))
        given.append(a)
    hypos = hypo(given)
    h = hypos["A2"]
    h = float(h)
    if h > 0.5:
        print(f"Your Cancer is Malignent with a {round(h*100,3)} % accuracy")
    elif h <= 0.5:
        print(f"Your Cancer is Benign with a {round((1-h)*100,3)} % accuracy")

def func(alpha, itera):
    scatter_plt()
    plt.show()
    for i in range(1,itera+1):
        algo(alpha)
        if i%(itera/10) == 0:    
            print(f"J = {J}\n")
            print(f"w2 = {w2}\n")
            print(f"b2 = {b2}\n")
    
    cost_plt(itera)
    line_eqn()
    print(f"The final Cost: {J}\n")
    print(f"The final W2: {w2}\n")
    print(f"The final b2: {b2}\n")
    res_train = accuracy(X_train,y_train)
    res_test = accuracy(X_test,y_test)
    res_train_raw = res_train["Raw"]
    res_train_clean = res_train["Clean"]
    res_test_raw = res_test["Raw"]
    res_test_clean = res_test["Clean"]
    print('\033[1m'+ colored(f"Accuracy of the raw training model: {round(res_train_raw * 100,3)}%","red"))
    print('\033[1m'+ colored(f"Accuracy of the raw testing model: {round(res_test_raw * 100,3)}%","red"))
    print('\033[1m'+ colored(f"Accuracy of the clean training model: {round(res_train_clean * 100,3)}%","red"))
    print('\033[1m'+ colored(f"Accuracy of the clean testing model: {round(res_test_clean * 100,3)}%","red"))
    x = input("\nDo you want to predict your own data? (y/n): ")
    if x.lower() == "y":
        predict()


func(0.000086,20000)