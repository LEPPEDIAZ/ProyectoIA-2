import matplotlib.pyplot as plt
import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoid3(Z):
    s = 1. / ( 1 + np.exp(-Z))
    return s

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

def obtener_costo(AL, y):
    m = y.shape[1]              
    costo = - (1 / m) * np.sum(
        np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))

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

def predict4(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid3(np.dot(w.T, X) + b)
    [print(x) for x in A]
    return A

def optimizar(w, b, X, Y, num_iterations, learning_rate, print_costo = False):
    costos = []
    
    for i in range(num_iterations):
    
        grads, costo = propagate2(w=w, b=b, X=X, Y=Y)
  
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b -  learning_rate*db
  
        if i % 100 == 0:
            costos.append(costo)
   
        if print_costo and i % 100 == 0:
            print (" %i: %f" %(i, costo))
    
    parametros = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return parametros, grads, costos