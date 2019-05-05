import matplotlib.pyplot as plt
import numpy as np


def initeparametros(dimensiones):
    np.random.seed(1)               
    parametros = {}
    L = len(dimensiones)            

    for l in range(1, L):           
        parametros["W" + str(l)] = np.random.randn(
            #0.01
            dimensiones[l], dimensiones[l - 1]) * 10
        parametros["b" + str(l)] = np.zeros((dimensiones[l], 1))

        assert parametros["W" + str(l)].shape == (
            dimensiones[l], dimensiones[l - 1])
        assert parametros["b" + str(l)].shape == (dimensiones[l], 1)

    return parametros
def initeparametros2(dimensiones):
    parametros = {}
    L = len(dimensiones)            
 
    for l in range(1, L):
        parametros['W' + str(l)] = np.random.randn(dimensiones[l], dimensiones[l-1]) / np.sqrt(dimensiones[l-1])
        parametros['b' + str(l)] = np.zeros((dimensiones[l], 1))
 
        assert(parametros['W' + str(l)].shape == (dimensiones[l], dimensiones[l-1]))
        assert(parametros['b' + str(l)].shape == (dimensiones[l], 1))
 
    return parametros
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoid2(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache
def sigmoid3(Z):
    s = 1. / ( 1 + np.exp(-Z))
    return s
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ
def propagate(w, b, X, Y):
    m = X.shape[1]
    #foward
    A = sigmoid(np.dot(w.T, X) + b)             
    cost = (-1 / m) * np.sum((Y*np.log(A) + np.diff(Y,1)*np.log(np.diff(A,1))), axis=1)
    #backward 
    X = np.transpose(X) 
    Y = np.transpose(Y)  
    dw = (1/m)*np.dot(X,((A-Y).T))
    db = (1/m)*np.sum(A-Y, axis=1)
    print(dw.shape)
    #assert(dw.shape == w.shape)
    #assert(db.dtype == float)
    cost = np.squeeze(cost)
    #assert(cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost
def propagate2(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid3(np.dot(w.T, X) + b)              
    cost = (-1. / m) * np.sum((Y*np.log(A) + (1 - Y)*np.log(1-A)), axis=1)    
    dw = (1./m)*np.dot(X,((A-Y).T))
    db = (1./m)*np.sum(A-Y, axis=1)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def tanh(Z):
    A = np.tanh(Z)
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z
def relu2(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache
def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, Z

def feed_forward(anterior, W, b):
    Z = np.dot(W, anterior) + b
    cache = (anterior, W, b)

    return Z, cache

def feed_act_forward(anterior, W, b, activacion):
   
    assert activacion == "sigmoid" or activacion == "tanh" or \
        activacion == "relu"

    if activacion == "sigmoid":
        Z, linear_cache = feed_forward(anterior, W, b)
        A, activation_cache = sigmoid(Z)

    elif activacion == "tanh":
        Z, linear_cache = feed_forward(anterior, W, b)
        A, activation_cache = tanh(Z)

    elif activacion == "relu":
        Z, linear_cache = feed_forward(anterior, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], anterior.shape[1])

    cache = (linear_cache, activation_cache)

    return A, cache
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
 
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
 
    return Z, cache
def feed_foward_model(X, parametros, theta="relu"):
    
    A = X                           
    caches = []                     
    L = len(parametros) // 2        

    for l in range(1, L):
        anterior = A
        A, cache = feed_act_forward(
            anterior, parametros["W" + str(l)], parametros["b" + str(l)],
            activacion=theta)
        caches.append(cache)

    AL, cache = feed_act_forward(
        A, parametros["W" + str(L)], parametros["b" + str(L)],
        activacion="sigmoid")
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches
def L_model_forward(X, parametros):
    caches = []
    A = X
    L = len(parametros) // 2                  

    for l in range(1, L):
        A_prev = A
        w_l = parametros['W' + str(l)]
        b_l = parametros['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, w_l, b_l, activation = "relu2")
        caches.append(cache)
 
    w_L = parametros['W' + str(L)]
    b_L = parametros['b' + str(L)]
    Yhat, cache = linear_activation_forward(A, w_L, b_L, activation = "sigmoid2")
    caches.append(cache)
 
    assert(Yhat.shape == (1,X.shape[1]))
 
    return Yhat, caches
def obtener_costo(AL, y):
    m = y.shape[1]              
    costo = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))

    return costo
def calcular_costo(Yhat, Y):
    m = Y.shape[1]
    logprobs = np.dot(Y, np.log(Yhat).T) + np.dot((1-Y), np.log(1-Yhat).T)
    costo = (-1./m) * logprobs 
    costo = np.squeeze(costo)      
    assert(costo.shape == ())
 
    return costo

def gradiente_de_sigmoid(thetaA, Z):
    A, Z = sigmoid(Z)
    dZ = thetaA * A * (1 - A)

    return dZ

def gradiente_de_tanh(thetaA, Z): 
    A, Z = tanh(Z)
    dZ = thetaA * (1 - np.square(A))

    return dZ

def gradiente_relu(thetaA, Z):
    
    A, Z = relu(Z)
    dZ = np.multiply(thetaA, np.int64(A > 0))

    return dZ
def linear_activation_forward(A_prev, W, b, activation):
    
    Z, linear_cache = linear_forward(A_prev, W, b)
 
    if activation == "sigmoid2":
        A, activation_cache = sigmoid2(Z)
 
    elif activation == "relu2":
        A, activation_cache = relu2(Z)
 
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
 
    return A, cache
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
 
    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
 
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
 
    return dA_prev, dW, db
def backpropagation(dZ, cache):
    
    anterior, W, b = cache
    m = anterior.shape[1]

    dW = (1 / m) * np.dot(dZ, anterior.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    danterior = np.dot(W.T, dZ)

    assert danterior.shape == anterior.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return danterior, dW, db


def act_backpropagation(thetaA, cache,activacion):
    
    linear_cache, activation_cache = cache

    if activacion == "sigmoid":
        dZ = gradiente_de_sigmoid(thetaA, activation_cache)
        danterior, dW, db = backpropagation(dZ, linear_cache)

    elif activacion == "tanh":
        dZ = gradiente_de_tanh(thetaA, activation_cache)
        danterior, dW, db = backpropagation(dZ, linear_cache)

    elif activacion == "relu":
        dZ = gradiente_relu(thetaA, activation_cache)
        danterior, dW, db = backpropagation(dZ, linear_cache)

    return danterior, dW, db


def modelo_backpropagation(AL, y, caches, theta="relu"):
    
    y = y.reshape(AL.shape)
    L = len(caches)
    grados = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    grados["thetaA" + str(L - 1)], grados["dW" + str(L)], grados[
        "db" + str(L)] = act_backpropagation(
            dAL, caches[L - 1], "sigmoid")

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grados["thetaA" + str(l - 1)], grados["dW" + str(l)], grados[
            "db" + str(l)] = act_backpropagation(
                grados["thetaA" + str(l)], current_cache,
                theta)

    return grados

def update_parametros(parametros, grados, curvaaprendisaje):
    
    L = len(parametros) // 2

    for l in range(1, L + 1):
        parametros["W" + str(l)] = parametros[
            "W" + str(l)] - curvaaprendisaje * grados["dW" + str(l)]
        parametros["b" + str(l)] = parametros[
            "b" + str(l)] - curvaaprendisaje * grados["db" + str(l)]

    return parametros
def predict(X, y, parametros):
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    a3, caches = feed_foward_model(X, parametros)
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("predict1: "  + str(np.mean((p[0,:] == y[0,:]))))
    return p
def predict2( X, y, parametros):
    m = X.shape[1]
    n = len(parametros)
    p = np.zeros((1,m))
    probas, caches = feed_foward_model(X, parametros)
    for i in range(0, probas.shape[1]):
        #if probas[0,i] &amp;&gt; 0.5:
        if probas[0,i] and probas > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("predict2: "  + str(np.sum((p == y)/m)))
 
    return p
def predict3(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
  
    A = sigmoid3(np.dot(w.T, X) + b)
 
    
    [print(x) for x in A]
    for i in range(A.shape[1]):
        
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
            
        else:
            Y_prediction[0, i] = 0
     
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def optimizar(w, b, X, Y, num_iterations, learning_rate, print_costo = False):
    costos = []
    
    for i in range(num_iterations):
    
        grads, costo = propagate(w=w, b=b, X=X, Y=Y)
  
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b -  learning_rate*db
  
        if i % 100 == 0:
            costos.append(costo)
   
        if print_costo and i % 100 == 0:
            print ("Costo despues de cada iteracion %i: %f" %(i, costo))
    
    parametros = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return parametros, grads, costos