import os as os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../functions/")
from load_dataset import load_dataset
from load_dataset1 import load_dataset1
from redesneuronales import (initeparametros,optimizar,sigmoid_backward,propagate,propagate2,linear_backward,L_model_forward,initeparametros2,calcular_costo,linear_forward,linear_activation_forward,sigmoid,sigmoid2,tanh,relu,relu2,leaky_relu,feed_forward, feed_act_forward, feed_foward_model,obtener_costo,gradiente_de_sigmoid,gradiente_de_tanh, gradiente_relu, backpropagation,act_backpropagation, modelo_backpropagation, update_parametros, predict, predict2,predict3)
import scipy
from PIL import Image
from scipy import ndimage
import skimage

# se importa el data
X_circulo, y_circulo = load_dataset("data2/")
index_dibujo = np.argmax(y_circulo); index_circulo = np.argmin(y_circulo)
#loading data
#circulo_X_orig=np.array(X_circulo["circulo_X_orig"][:])
#circulo_y_orig=np.array(y_circulo["circulo_y_orig"][:])
#circulo_set_y = y_circulo.reshape((1, y_circulo.shape[0]))

#se importa data del circulo
X_test, y_test = load_dataset("dibujo/")
index_dibujo = np.argmax(y_test); index_dibujo = np.argmin(y_test)

#se importa cuadrado
X_cuadrado, y_cuadrado = load_dataset1("data2/")
index_dibujo = np.argmax(y_cuadrado); index_cuadrado = np.argmin(y_cuadrado)
#resshape origin and examples
circulo_set_x_flatten = X_circulo.reshape(X_circulo.shape[0], -1).T
cuadrado_set_x_flatten = X_cuadrado.reshape(X_cuadrado.shape[0], -1).T
test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T
#se tienen 5 capas, input, output y 3 capas escondidas
#constantes que definen el modelo
#se define la estructura de la red neuronal
n_x_circulo = circulo_set_x_flatten.shape[0]
n_x_cuadrado = cuadrado_set_x_flatten.shape[0]
n_x_test = test_set_x_flatten.shape[0]
num_px=150
n_y = 1 
nn_layers_circulo = [n_x_circulo, 20, 7, 5, n_y]
nn_layers_cuadrado = [n_x_cuadrado, 20, 7, 5, n_y]
print("Capas del circulo y cuadrados")
print (nn_layers_circulo)
nn_layers_cuadrado = [n_x_cuadrado, 20, 7, 5, n_y]
print (nn_layers_cuadrado)

print(f"""Dimensiones originales para el circulo:\n{20 * '-'}\nData de prueba: {X_cuadrado.shape}, {y_cuadrado.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el cuadrado:\n{20 * '-'}\nData de prueba: {X_circulo.shape}, {y_circulo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print ("circulo_set_x_flatten shape: " + str(circulo_set_x_flatten.shape))
print ("cuadrado_set_x_flatten shape: " + str(cuadrado_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

#debido a que se necesita RBG cada pixel es un vector de numeros de 0 a 255.
X_cuadrado = X_cuadrado / 255
X_circulo = X_circulo / 255
X_test = X_test / 255
len_cuadrado=len(X_cuadrado)
len_circulo=len(X_circulo)
len_test=len(X_test)
print(len_cuadrado)
print(len_circulo)
print(len_test)
test_dataset=X_circulo, y_circulo
classes = np.array(test_dataset)
index = 25
example = X_circulo[index]
circulo = "circulo"
print ("y = " + str(y_circulo[:, index]) + ", es una foto de un " + circulo)
#print ("y = " + str(y_circulo[:, index]) + ",es una foto de un  '" + classes[np.squeeze(y_circulo[:, index])].decode("utf-8") )

def inicializar_parametros_2(dimensiones, initialization_method="he"):
    np.random.seed(1)               
    parametros = {}                 
    L = len(dimensiones)            

    if initialization_method == "he":
        for l in range(1, L):
            parametros["W" + str(l)] = np.random.randn(
                dimensiones[l],
                dimensiones[l - 1]) * np.sqrt(2 / dimensiones[l - 1])
            parametros["b" + str(l)] = np.zeros((dimensiones[l], 1))
    elif initialization_method == "xavier":
        for l in range(1, L):
            parametros["W" + str(l)] = np.random.randn(
                dimensiones[l],
                dimensiones[l - 1]) * np.sqrt(1 / dimensiones[l - 1])
            parametros["b" + str(l)] = np.zeros((dimensiones[l], 1))

    return parametros
       
def inicializar_parametros_cero(dimensiones):
    np.random.seed(1)               
    parametros = {}                 
    L = len(dimensiones)            

    for l in range(1, L):
        parametros["W" + str(l)] = np.zeros(
            (dimensiones[l], dimensiones[l - 1]))
        parametros["b" + str(l)] = np.zeros((dimensiones[l], 1))

    return parametros
def inicializarconcero(dimensiones):
    w = np.zeros(shape=(dimensiones, 1), dtype=np.float32)
    b = 0
    assert(w.shape == (dimensiones, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b
dim = 2
w, b = inicializarconcero(dim)
print ("w = " + str(w))
print ("b = " + str(b))
w, b, X, y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate2(w, b, X, y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
print("-----------------------------------")
params, grads, costs = optimizar(w, b, X, y, num_iterations= 10, learning_rate = 0.009, print_costo = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("-----------------------------------")
print ("predictions = " + str(predict3(w, b, X)))
def gradiente_aprendisaje(
        X, y, dimensiones, curvaaprendisaje=0.01, num_iterations=500,
        print_costo=True, theta="relu", initialization_method="he" ):
   
    np.random.seed(1)

  
    #parametros = initeparametros(dimensiones)

    costo_list = []

    if initialization_method == "zeros":
        parametros = inicializar_parametros_cero(dimensiones)
    elif initialization_method == "random":
        parametros = initialize_parameters_random(dimensiones)
    else:
        parametros = inicializar_parametros_2(
            dimensiones, initialization_method)


    for i in range(num_iterations):
        
        AL, caches = feed_foward_model(
            X, parametros, theta)
        costo = obtener_costo(AL, y)
        grados = modelo_backpropagation(AL, y, caches, theta)
        parametros = update_parametros(parametros, grados, curvaaprendisaje)
        if (i + 1) % 100 == 0 and print_costo:
            print(f"El costo despues de  {i + 1} interaciones es: {costo:.4f}")

        if i % 100 == 0:
            costo_list.append(costo)


    return parametros


def accuracy(X, parametros, y,activacion="relu"):
    
    probs, caches = feed_foward_model(X, parametros, activacion)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100

    return print (f"El accuracy es: {accuracy:.2f}%.")
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = inicializarconcero(X_train.shape[0])
    parameters, grads, costs = optimizar(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
   
    w = parameters["w"]
    b = parameters["b"]
    

    Y_prediction_test = predict3(w, b, X_test)
    Y_prediction_train = predict3(w, b, X_train)

    print("data accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))


    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
circuloread = model(X_circulo, y_circulo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
print("Circulo Reading...")
my_image = "test.dibujo.jpg"  
fname = "dibujo/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict3(circuloread["w"], circuloread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")
cuadradoread = model(X_cuadrado,y_cuadrado, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("Cuadrado Reading...")
my_predicted_image = predict3(cuadradoread["w"], cuadradoread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")
print("se calcula el circulo")

dimensiones = [X_circulo.shape[0], 5, 5, 1]

parametros_relu = gradiente_aprendisaje(
    X_circulo, y_circulo, dimensiones, curvaaprendisaje=0.03, num_iterations=500,
    theta="relu")

accuracy(X_circulo, parametros_relu, y_circulo, activacion="relu")
accuracy(X_test, parametros_relu, y_test, activacion="relu")
pred_train = predict(X_test, y_test, parametros_relu)
pred_train = predict2(X_test, y_test, parametros_relu)

parametros_tanh = gradiente_aprendisaje(X_circulo, y_circulo, dimensiones, theta="tanh",
                   initialization_method="he")

accuracy(X_circulo, parametros_tanh,y_circulo, "tanh")
accuracy(X_test, parametros_tanh,y_test, "tanh")
pred_train = predict(X_test, y_test, parametros_tanh)
pred_train = predict2(X_test, y_test, parametros_relu)
print("-------------------------------------------------------------------")
print("se calcula el cuadrado")

dimensiones = [X_cuadrado.shape[0], 5, 5, 1]

parametros_relu = gradiente_aprendisaje(
    X_cuadrado, y_cuadrado, dimensiones, curvaaprendisaje=0.03, num_iterations=500,
    theta="relu")

accuracy(X_cuadrado, parametros_relu, y_cuadrado, activacion="relu")
accuracy(X_test, parametros_relu, y_test, activacion="relu")
pred_train = predict(X_test, y_test, parametros_relu)
pred_train = predict2(X_test, y_test, parametros_relu)
parametros_tanh = gradiente_aprendisaje(X_cuadrado, y_cuadrado, dimensiones, theta="tanh",
                   initialization_method="he")

accuracy(X_cuadrado, parametros_tanh, y_cuadrado, "tanh")
accuracy(X_test, parametros_tanh,y_test, "tanh")
pred_train = predict(X_test, y_test, parametros_tanh)
pred_train = predict2(X_test, y_test, parametros_tanh)
