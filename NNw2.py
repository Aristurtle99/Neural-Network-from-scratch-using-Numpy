import numpy as np
import matplotlib.pyplot as plt



def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    params = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return params

def predict(X,params):
    array = np.array([])
    A2, _ = forward_prop(X, params)

    for i in range(X.shape[1]):
        if(A2[0,i] > 0.5):
            array = np.append(array, 1)
        else:
            array = np.append(array, 0)
    return array
        
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def forward_prop(X,params):
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)

    cache = {'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2}
    return A2,cache

def compute_cost(A2,Y):
    m = Y.shape[1]
    L = (1/m)*np.sum(-( Y*np.log(A2) + (1 - Y)*np.log(1 -A2) ))
    return L

def backward_prop(params,cache,X,Y):
   
    W2 = params['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis = 1, keepdims = True)
    dA1 = np.dot(W2.T,dZ2)
   
    dZ1 = dA1 * (1-A1**2)
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1,axis = 1, keepdims = True)

    
    grads = {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
    return grads
def update_params(params,grads,learning_rate):


    dw1 = grads['dW1']
    db1 = grads['db1']
    dw2 = grads['dW2']
    db2 = grads['db2']

    params['W1'] -= learning_rate*dw1
    params['b1'] -= learning_rate*db1
    params['W2'] -= learning_rate*dw2
    params['b2'] -= learning_rate*db2

    return params

np.random.seed(0)
X = np.random.randn(2, 400)                              
Y = (X[0]**2 + X[1]**2 < 1).astype(int).reshape(1, 400)  

def train(X,Y,n_h,iters,rate):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    loss = []
    params = initialize_parameters(n_x,n_h,n_y)

    for i in range(iters):
        A2,cache = forward_prop(X,params)
        L = compute_cost(A2,Y)
        grads = backward_prop(params,cache,X,Y)
        params = update_params(params,grads,rate)

        if i%1000 == 0:
            print("Iteration %d: L = %.4f" % (i,L))
        if i%100 == 0:
            loss.append(L)

    return params, loss

params, loss = train(X,Y,n_h = 4,iters = 10000,rate = 0.5)
preds = predict(X, params)
print("accuracy:", np.mean(preds == Y))


plt.plot(range(0, 10000, 100), loss)
plt.xlabel("iteration")
plt.ylabel("cost L")
plt.title("Training loss")
plt.show()

    
