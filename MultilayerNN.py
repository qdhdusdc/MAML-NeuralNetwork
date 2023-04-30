import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.transforms import Bbox

a1 = np.random.uniform(0.1, 1)#, size=(num_tasks, 1))
b1 = np.random.uniform(-np.pi, np.pi)#, size=(num_tasks, 1))
lb,ub=-5,5
lr=0.00001

input_dim=5
hidden1_dim=20
out_dim=input_dim
n_sample=1

epoches = 500
tasks = 100
beta = 0.01


def samplePoints(k,test=None):#flag=1 for maml training, a,b will update in every iteration. Else a,b are global variables for test.
    x = (ub-lb)*np.random.rand(k,input_dim)+lb
    if test==True:
        y = a1 * np.sin(x + b1)
        return x,y
    a = np.random.uniform(0.1, 1)#, size=(num_tasks, 1))
    b = np.random.uniform(-np.pi, np.pi)#, size=(num_tasks, 1))
    y = a * np.sin(x + b)
    return x,y

class MamlModel:
    def __init__(self,input_dim,hidden1_dim,out_dim,n_sample):
        super(MamlModel, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.W1 = np.zeros((input_dim,hidden1_dim))
        self.W2 = np.zeros((hidden1_dim,out_dim))
        self.b1 = np.zeros((hidden1_dim,))
        self.b2 = np.zeros((out_dim,))

    def forward(self):
        X_train,Y_train = samplePoints(self.n_sample)
        hidden1 = np.dot(X_train,self.W1) + self.b1
        Y_predict = np.dot(hidden1,self.W2) + self.b2
        return X_train,Y_train,Y_predict

def loss_function(y_pred,y_true):
    result=0
    for i in range(0,len(y_pred)):
        result+=(y_pred[0][i]-y_true[0][i])**2
    return 0.5*result

maml = MamlModel(input_dim,hidden1_dim,out_dim,n_sample)
maml_W1 = np.random.rand(input_dim,hidden1_dim)
maml_W2 = np.random.rand(hidden1_dim,out_dim)
maml_b1 = np.random.rand(hidden1_dim,)
maml_b2 = np.random.rand(out_dim)

train_maml_W1 = np.random.rand(tasks,input_dim,hidden1_dim)
train_maml_W2 = np.random.rand(tasks,hidden1_dim,out_dim)
train_maml_b1 = np.random.rand(tasks,hidden1_dim,)
train_maml_b2 = np.random.rand(tasks,out_dim)

meta_gradient_W1=np.random.rand(input_dim,hidden1_dim)
meta_gradient_W2=np.random.rand(hidden1_dim,out_dim)
meta_gradient_b1=np.random.rand(hidden1_dim,)
meta_gradient_b2=np.random.rand(out_dim)

def SimpleLayerNN(x,y,epoch,lr):
    global maml_W1,maml_W2,maml_b1,maml_b2
    W1,W2,b1,b2=maml_W1.copy(),maml_W2.copy(),maml_b1.copy(),maml_b2.copy()
    h = np.dot(x,W1)+b1
    Y_predict = np.dot(h,W2)+b2
    
    for i in range(1,epoch):
        h = np.dot(x,W1)+b1
        Y_predict = np.dot(h,W2)+b2
        loss = loss_function(y,Y_predict)
        if i%100 == 0:
            print("Epoch:{:d}".format(i), "Loss:%s"%loss)
        db2 = Y_predict-y
        db2 = db2[0]
        db1 = np.dot((Y_predict-y),W2.T)
        db1 = db1[0]
        dW1 = np.dot(x.T,np.dot((Y_predict-y),W2.T))
        dW2 = np.dot(h.T,(Y_predict-y))
        
        W1-=lr*dW1
        W2-=lr*dW2
        b1-=lr*db1
        b2-=lr*db2
    return x,Y_predict

def SimpleLayerNN_Random_Parameter(x,y,epoch,lr,W_shape):
    
    W1 = np.random.rand(input_dim,hidden1_dim)
    W2 = np.random.rand(hidden1_dim,out_dim)
    b1 = np.random.rand(hidden1_dim,)
    b2 = np.random.rand(out_dim)
    h = np.dot(x,W1)+b1
    Y_predict = np.dot(h,W2)+b2
    
    for i in range(1,epoch):
        h = np.dot(x,W1)+b1
        Y_predict = np.dot(h,W2)+b2
        loss = loss_function(y,Y_predict)
        if i%100 == 0:
            print("Epoch:{:d}".format(i), "Loss:%s"%loss)
        db2 = Y_predict-y
        db2 = db2[0]
        db1 = np.dot((Y_predict-y),W2.T)
        db1 = db1[0]
        dW1 = np.dot(x.T,np.dot((Y_predict-y),W2.T))
        dW2 = np.dot(h.T,(Y_predict-y))
        
        W1-=lr*dW1
        W2-=lr*dW2
        b1-=lr*db1
        b2-=lr*db2
    return x,Y_predict

#maml training
def train(epoch):
    #Training on each task and retain the parameters
    global meta_gradient_W1,meta_gradient_W2,meta_gradient_b1,meta_gradient_b2
    global maml_W1,maml_W2,maml_b1,maml_b2,train_maml_W1,train_maml_W2,train_maml_b1,train_maml_b2
    loss_sum = 0.0
    for i in range(tasks):
        maml.W1,maml.W2,maml.b1,maml.b2 = maml_W1,maml_W2,maml_b1,maml_b2
        X_train, Y_train, Y_predict = maml.forward()
        loss_value = loss_function(Y_train, Y_predict)
        loss_sum = loss_sum + loss_value
        
        h = np.dot(X_train,maml.W1)+maml.b1
        db2 = Y_predict-Y_train
        db2 = db2[0]
        db1 = np.dot((Y_predict-Y_train),maml.W2.T)
        db1 = db1[0]
        dW1 = np.dot(X_train.T,np.dot((Y_predict-Y_train),maml.W2.T))
        dW2 = np.dot(h.T,(Y_predict-Y_train))
        
        maml.W1-=lr*dW1
        maml.W2-=lr*dW2
        maml.b1-=lr*db1
        maml.b2-=lr*db2
        
        train_maml_W1[i] = maml.W1
        train_maml_W2[i] = maml.W2
        train_maml_b1[i] = maml.b1
        train_maml_b2[i] = maml.b2
        
    # testing on each task
    for i in range(tasks):
        train_maml_W1[i] = maml.W1
        train_maml_W2[i] = maml.W2
        train_maml_b1[i] = maml.b1
        train_maml_b2[i] = maml.b2

        h = np.dot(X_train,maml.W1)+maml.b1
        db2 = Y_predict-Y_train
        db2 = db2[0]
        db1 = np.dot((Y_predict-Y_train),maml.W2.T)
        db1 = db1[0]
        dW1 = np.dot(X_train.T,np.dot((Y_predict-Y_train),maml.W2.T))
        dW2 = np.dot(h.T,(Y_predict-Y_train))
        
        
        maml.W1-=lr*dW1
        maml.W2-=lr*dW2
        maml.b1-=lr*db1
        maml.b2-=lr*db2
                
        X_test, Y_test, Y_predict_test = maml.forward()
        loss_value = loss_function(Y_test, Y_predict_test)
        
        
        meta_gradient_W1 = meta_gradient_W1 + maml.W1
        meta_gradient_W2 = meta_gradient_W2 + maml.W2
        meta_gradient_b1 = meta_gradient_b1 + maml.b1
        meta_gradient_b2 = meta_gradient_b2 + maml.b2
        
    maml.W1 -= beta * meta_gradient_W1 / tasks
    maml.W2 -= beta * meta_gradient_W2 / tasks
    maml.b1 -= beta * meta_gradient_b1 / tasks
    maml.b2 -= beta * meta_gradient_b2 / tasks
    if  epoch%100==0:
        print("the Epoch is {:04d}".format(epoch),"the Loss is {:.4f}".format(loss_sum/tasks))

def fit_func(x,a,b):
    return a*np.sin(x+b)

if __name__ == "__main__":
    for epoch in range(epoches):
        train(epoch)
    x,y = samplePoints(1,test=True)

    plt.scatter(x, y, marker='^',s=25,alpha=0.7)
    x_true = np.linspace(lb, ub, 200)
    y_true = [a1*np.sin(xi+b1) for xi in x_true]
    x1,y1 = SimpleLayerNN(x,y,0,lr)
    x2,y2 = SimpleLayerNN(x,y,3,lr)
    x3,y3 = SimpleLayerNN(x,y,10,lr)
    params1, params_covariance1 = curve_fit(fit_func, x1[0], y1[0])
    params2, params_covariance2 = curve_fit(fit_func, x2[0], y2[0])
    params3, params_covariance3 = curve_fit(fit_func, x3[0], y3[0])
    xy_sorted = sorted(zip(x1[0], y1[0]))
    x1 = [e[0] for e in xy_sorted]
    y1 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(x2[0], y2[0]))
    x2 = [e[0] for e in xy_sorted]
    y2 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(x3[0], y3[0]))
    x3 = [e[0] for e in xy_sorted]
    y3 = [e[1] for e in xy_sorted]      
    
    x4,y4 = SimpleLayerNN_Random_Parameter(x,y,0,lr,[input_dim,input_dim])
    x5,y5 = SimpleLayerNN_Random_Parameter(x,y,3,lr,[input_dim,input_dim])
    x6,y6 = SimpleLayerNN_Random_Parameter(x,y,10,lr,[input_dim,input_dim])
    params4, params_covariance4 = curve_fit(fit_func, x4[0], y4[0])
    params5, params_covariance5 = curve_fit(fit_func, x5[0], y5[0])
    params6, params_covariance6 = curve_fit(fit_func, x6[0], y6[0])
    xy_sorted = sorted(zip(x4[0], y4[0]))
    x4 = [e[0] for e in xy_sorted]
    y4 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(x5[0], y5[0]))
    x5 = [e[0] for e in xy_sorted]
    y5 = [e[1] for e in xy_sorted]
    xy_sorted = sorted(zip(x6[0], y6[0]))
    x6 = [e[0] for e in xy_sorted]
    y6 = [e[1] for e in xy_sorted]      
    
    y1 = [params1[0]*np.sin(xi+params1[1]) for xi in x_true]
    y2 = [params2[0]*np.sin(xi+params2[1]) for xi in x_true]
    y3 = [params3[0]*np.sin(xi+params3[1]) for xi in x_true]
    y4 = [params4[0]*np.sin(xi+params4[1]) for xi in x_true]
    y5 = [params5[0]*np.sin(xi+params5[1]) for xi in x_true]
    y6 = [params6[0]*np.sin(xi+params6[1]) for xi in x_true]
    
    plt.plot(x_true,y_true,color='red', label='True Function')
    plt.plot(x_true,y1,color='yellow', label='After 0 Steps(With maml)')
    plt.plot(x_true,y2,color='green', label='After 3 Steps(With maml)')
    plt.plot(x_true,y3,color='blue', label='After 10 Steps(With maml)')
    plt.plot(x_true,y4,color='yellow', linestyle='--',label='After 0 Steps(Without maml)')
    plt.plot(x_true,y5,color='green', linestyle='--',label='After 3 Steps(Without maml)')
    plt.plot(x_true,y6,color='blue', linestyle='--',label='After 10 Steps(Without maml)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right', bbox_to_anchor=(2, 1))
    plt.savefig("MultilayerNN.png",dpi=300,bbox_inches=Bbox.from_bounds(*(0,0,10,6)))
    plt.show()
    plt.close('all')
    