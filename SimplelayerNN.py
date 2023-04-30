import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.transforms import Bbox

a1 = np.random.uniform(0.1, 1)#, size=(num_tasks, 1))
b1 = np.random.uniform(-np.pi, np.pi)#, size=(num_tasks, 1))
lb,ub=-5,5
input_dim=5
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
    a = np.random.uniform(0.1, 1)
    b = np.random.uniform(-np.pi, np.pi)
    y = a * np.sin(x + b)
    return x,y

class MamlModel:
    def __init__(self,input_dim,out_dim,n_sample):
        super(MamlModel, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.W = np.zeros((input_dim,out_dim))
    def forward(self):
        X_train,Y_train = samplePoints(self.n_sample)
        Y_predict = np.dot(X_train,self.W)
        return X_train,Y_train,Y_predict

maml = MamlModel(input_dim,out_dim,n_sample)
train_maml_W = np.random.rand(tasks,input_dim,input_dim) #weight for each task
maml_W = np.random.rand(input_dim,input_dim) #final result
meta_gradient = np.random.rand(input_dim,input_dim)

def SimpleLayerNN(x,y,epoch,lr):
    global maml_W
    W = maml_W.copy()
    Y_predict = np.dot(x,W)
    for i in range(1,epoch):
        dW = np.dot(x.T,(Y_predict-y))
        Y_predict = np.dot(x,W)
        loss = loss_function(y,Y_predict)
        if i%100 == 0:
            print("Epoch:{:d}".format(i), "Loss:%s"%loss)
        W-=dW
    return x,Y_predict

def SimpleLayerNN_Random_Parameter(x,y,epoch,lr,W_shape):
    W = np.random.rand(input_dim,out_dim)
    Y_predict = np.dot(x,W)
    for i in range(1,epoch):
        dW = np.dot(x.T,(Y_predict-y))
        Y_predict = np.dot(x,W)
        loss = loss_function(y,Y_predict)
        if i%100 == 0:
            print("Epoch:{:d}".format(i), "Loss:%s"%loss)
        W-=dW
    return x,Y_predict

def loss_function(y_pred,y_true):
    result=0
    for i in range(0,len(y_pred)):
        result+=(y_pred[0][i]-y_true[0][i])**2
    return 0.5*result

def train(epoch):
    global maml_W,meta_gradient
    loss_sum = 0.0
    for i in range(tasks):
        maml.W = maml_W
        X_train, Y_train, Y_predict = maml.forward()
        loss_value = loss_function(Y_train, Y_predict)
        loss_sum = loss_sum + loss_value
        train_maml_W[i] = maml.W

    for i in range(tasks):
        maml.W = train_maml_W[i]
        X_train,Y_test, Y_predict_test = maml.forward()
        loss_value = loss_function(Y_test, Y_predict_test)
        meta_gradient = meta_gradient + maml.W

    maml_W = maml_W - beta * meta_gradient / tasks
    if  epoch%100==0:
        print("the Epoch is {:04d}".format(epoch),"the Loss is {:.4f}".format(loss_sum/tasks))

def fit_func(x,a,b):
    return a*np.sin(x+b)

if __name__ == "__main__":
    
    for epoch in range(epoches):
        train(epoch)
    x,y = samplePoints(1,test=True)
    
    lr=0.001
    
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
    plt.savefig("SimplelayerNN.png",dpi=300,bbox_inches=Bbox.from_bounds(*(0,0,11,5)))
    plt.show()
    plt.close('all')