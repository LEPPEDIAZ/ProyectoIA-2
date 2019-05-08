import sys
from matplotlib import pyplot as plt
import numpy as np
sys.path.append("../functions/")
from load_dataset6 import load_dataset6
from load_dataset7 import load_dataset7
from load_dataset8 import load_dataset8
from load_dataset import load_dataset
from redesneuronales import (initeparametros,optimizar,sigmoid_backward,propagate,propagate2,linear_backward,L_model_forward,initeparametros2,calcular_costo,linear_forward,linear_activation_forward,sigmoid,sigmoid2,tanh,relu,relu2,leaky_relu,feed_forward, feed_act_forward, feed_foward_model,obtener_costo,gradiente_de_sigmoid,gradiente_de_tanh, gradiente_relu, backpropagation,act_backpropagation, modelo_backpropagation, update_parametros, predict, predict2,predict3)
import scipy
from PIL import Image
from scipy import ndimage
import skimage
import tkinter 
from tkinter import *

# se importa el data de la casa
X_casa, y_casa = load_dataset6("data4/")
index_dibujo = np.argmax(y_casa); index_casa = np.argmin(y_casa)

#se importa data del dibujo
X_test, y_test = load_dataset("dibujo/")
index_dibujo = np.argmax(y_test); index_dibujo = np.argmin(y_test)

#se importa el feliz
X_feliz, y_feliz = load_dataset7("data4/")
index_dibujo = np.argmax(y_feliz); index_feliz = np.argmin(y_feliz)

#se importa la cara triste
X_triste, y_triste = load_dataset8("data4/")
index_dibujo = np.argmax(y_feliz); index_triste = np.argmin(y_feliz)
#resshape origin and examples
casa_set_x_flatten = X_casa.reshape(X_casa.shape[0], -1).T
feliz_set_x_flatten = X_feliz.reshape(X_feliz.shape[0], -1).T
triste_set_x_flatten = X_triste.reshape(X_triste.shape[0], -1).T
test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T
#se tienen 5 capas, input, output y 3 capas escondidas
#constantes que definen el modelo
#se define la estructura de la red neuronal
n_x_casa = casa_set_x_flatten.shape[0]
n_x_feliz = feliz_set_x_flatten.shape[0]
n_x_triste = triste_set_x_flatten.shape[0]
n_x_test = test_set_x_flatten.shape[0]
num_px=150
n_y = 1 
nn_layers_casa = [n_x_casa, 20, 7, 5, n_y]
nn_layers_feliz = [n_x_feliz, 20, 7, 5, n_y]
nn_layers_triste = [n_x_triste, 20, 7, 5, n_y]
print("Capas de casa y feliz")
print (nn_layers_casa)
nn_layers_feliz = [n_x_feliz, 20, 7, 5, n_y]
print (nn_layers_feliz)
print (nn_layers_triste)
print(f"""Dimensiones originales para la cara feliz:\n{20 * '-'}\nData de prueba: {X_feliz.shape}, {y_feliz.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para la casa:\n{20 * '-'}\nData de prueba: {X_casa.shape}, {y_casa.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para la cara triste:\n{20 * '-'}\nData de prueba: {X_triste.shape}, {y_triste.shape}
Test: {X_test.shape}, {y_test.shape}""")
print ("casa_set_x_flatten shape: " + str(casa_set_x_flatten.shape))
print ("feliz_set_x_flatten shape: " + str(feliz_set_x_flatten.shape))
print ("triste_set_x_flatten shape: " + str(triste_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

#debido a que se necesita RBG cada pixel es un vector de numeros de 0 a 255.
X_feliz = X_feliz / 255
X_casa = X_casa / 255
X_triste = X_triste / 255
X_test = X_test / 255
len_feliz=len(X_feliz)
len_casa=len(X_casa)
len_test=len(X_test)
len_triste=len(X_triste)
print(len_feliz)
print(len_casa)
print(len_test)
print(len_triste)
test_dataset=X_casa, y_casa
classes = np.array(test_dataset)
index = 25
example = X_casa[index]
casa = "casa"
print ("y = " + str(y_casa[:, index]) + ", es una foto de un " + casa)


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
        X, y, dimensiones, curvaaprendisaje=0.01, num_iterations=200,
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

    return accuracy
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = inicializarconcero(X_train.shape[0])
    parameters, grads, costs = optimizar(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
   
    w = parameters["w"]
    b = parameters["b"]
    

    Y_prediction_test = predict3(w, b, X_test)
    Y_prediction_train = predict3(w, b, X_train)

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
casaread = model(X_casa, y_casa, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

print("-------------------------------------------------------------------")
felizread = model(X_feliz,y_feliz, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
tristeread = model(X_triste,y_triste, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")
print("se calcula la casa")

dimensiones = [X_casa.shape[0], 5, 5, 1]

parametros_tanh1 = gradiente_aprendisaje(X_casa, y_casa, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Casa Reading...")
my_image = "test.dibujo.jpg"  
fname = "dibujo/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_casa = predict3(casaread["w"], casaread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")

print("-------------------------------------------------------------------")
print("se calcula la cara feliz")



dimensiones = [X_feliz.shape[0], 5, 5, 1]

parametros_tanh2 = gradiente_aprendisaje(X_feliz, y_feliz, dimensiones, theta="tanh",
                   initialization_method="he")

print("Feliz Reading...")
my_predicted_feliz = predict3(felizread["w"], felizread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula la cara triste")

dimensiones = [X_triste.shape[0], 5, 5, 1]

parametros_tanh3 = gradiente_aprendisaje(X_triste, y_triste, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Triste Reading...")
my_predicted_triste = predict3(tristeread["w"], tristeread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")
print("casa")
casa_data_casa=accuracy(X_casa, parametros_tanh1,y_casa, "tanh")
test_data_test_casa=accuracy(X_test, parametros_tanh1,y_test, "tanh")
print ("test de feliz")
feliz_data_feliz=accuracy(X_feliz, parametros_tanh2, y_feliz, "tanh")
test_data_feliz_test=accuracy(X_test, parametros_tanh2,y_test, "tanh")
print("test de triste")
triste_data_triste=accuracy(X_triste, parametros_tanh3, y_feliz, "tanh")
test_data_triste_test=accuracy(X_test, parametros_tanh3,y_test, "tanh")
print("-------------------------------------------------------------------")
if(my_predicted_casa > my_predicted_triste) and (my_predicted_casa > my_predicted_feliz):
    largest=my_predicted_casa
    categoria="casa"
    print("casa es mayor")
    print (str(predict3(casaread["w"], casaread["b"], my_image)))
    print(casa_data_casa)
elif (my_predicted_casa < my_predicted_triste) and (my_predicted_casa < my_predicted_feliz):
    largest=my_predicted_triste
    categoria="triste"
    print("triste es mayor")
    print (str(predict3(tristeread["w"], tristeread["b"], my_image)))
    print(triste_data_triste)
else:
    largest=my_predicted_feliz
    categoria="feliz" 
    print("feliz es mayor")
    print("-")
    print (str(predict3(felizread["w"], felizread["b"], my_image)))
    print("-")
    print(feliz_data_feliz)
   
    




print("-------------------------------------------------------------------")
