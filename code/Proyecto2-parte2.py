import sys
from matplotlib import pyplot as plt
import numpy as np
sys.path.append("../functions/")
from load_dataset3 import load_dataset3
from load_dataset4 import load_dataset4
from load_dataset5 import load_dataset5
from load_dataset import load_dataset
from redesneuronales import (optimizar,propagate2,sigmoid,tanh,relu,feed_forward, feed_act_forward, feed_foward_model,obtener_costo,gradiente_de_sigmoid,gradiente_de_tanh, gradiente_relu, backpropagation,act_backpropagation, modelo_backpropagation, update_parametros, predict4)
import scipy
from PIL import Image
from scipy import ndimage
import skimage
import tkinter 
from tkinter import *

# se importa el data del arbol
X_arbol, y_arbol = load_dataset3("data3/")
index_dibujo = np.argmax(y_arbol); index_arbol = np.argmin(y_arbol)

#se importa data del dibujo
X_test, y_test = load_dataset("dibujo/")
index_dibujo = np.argmax(y_test); index_dibujo = np.argmin(y_test)

#se importa huevo
X_huevo, y_huevo = load_dataset4("data3/")
index_dibujo = np.argmax(y_huevo); index_huevo = np.argmin(y_huevo)

#se importa el interrogacion
X_interrogacion, y_interrogacion = load_dataset5("data3/")
index_dibujo = np.argmax(y_huevo); index_interrogacion = np.argmin(y_huevo)
#resshape origin and examples
arbol_set_x_flatten = X_arbol.reshape(X_arbol.shape[0], -1).T
huevo_set_x_flatten = X_huevo.reshape(X_huevo.shape[0], -1).T
interrogacion_set_x_flatten = X_interrogacion.reshape(X_interrogacion.shape[0], -1).T
test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T
#se tienen 5 capas, input, output y 3 capas escondidas
#constantes que definen el modelo
#se define la estructura de la red neuronal
n_x_arbol = arbol_set_x_flatten.shape[0]
n_x_huevo = huevo_set_x_flatten.shape[0]
n_x_interrogacion = interrogacion_set_x_flatten.shape[0]
n_x_test = test_set_x_flatten.shape[0]
num_px=150
n_y = 1 
nn_layers_arbol = [n_x_arbol, 20, 7, 5, n_y]
nn_layers_huevo = [n_x_huevo, 20, 7, 5, n_y]
nn_layers_interrogacion = [n_x_interrogacion, 20, 7, 5, n_y]
print("Capas del arbol y huevo")
print (nn_layers_arbol)
nn_layers_huevo = [n_x_huevo, 20, 7, 5, n_y]
print (nn_layers_huevo)
print (nn_layers_interrogacion)
print(f"""Dimensiones originales para el huevo:\n{20 * '-'}\nData de prueba: {X_huevo.shape}, {y_huevo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el arbol:\n{20 * '-'}\nData de prueba: {X_arbol.shape}, {y_arbol.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el interrogacion:\n{20 * '-'}\nData de prueba: {X_interrogacion.shape}, {y_interrogacion.shape}
Test: {X_test.shape}, {y_test.shape}""")
print ("arbol_set_x_flatten shape: " + str(arbol_set_x_flatten.shape))
print ("huevo_set_x_flatten shape: " + str(huevo_set_x_flatten.shape))
print ("interrogacion_set_x_flatten shape: " + str(interrogacion_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

#debido a que se necesita RBG cada pixel es un vector de numeros de 0 a 255.
X_huevo = X_huevo / 255
X_arbol = X_arbol / 255
X_interrogacion = X_interrogacion / 255
X_test = X_test / 255
len_huevo=len(X_huevo)
len_arbol=len(X_arbol)
len_test=len(X_test)
len_interrogacion=len(X_interrogacion)
print(len_huevo)
print(len_arbol)
print(len_test)
print(len_interrogacion)
test_dataset=X_arbol, y_arbol
classes = np.array(test_dataset)
index = 25
example = X_arbol[index]
arbol = "arbol"
print ("y = " + str(y_arbol[:, index]) + ", es una foto de un " + arbol)


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
print ("predictions = " + str(predict4(w, b, X)))
def gradiente_aprendisaje(
        X, y, dimensiones, curvaaprendisaje=0.01, num_iterations=200,
        print_costo=True, theta="relu", initialization_method="he" ):
   
    np.random.seed(1)

  


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
            print(f"{i + 1} : {costo:.4f}")

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
    

    Y_prediction_test = predict4(w, b, X_test)
    Y_prediction_train = predict4(w, b, X_train)

    d = {"costos": costs,
         "Y_pred_test": Y_prediction_test, 
         "Y_pred_variable" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "cantidad_iteraciones": num_iterations}
    
    return d
arbolread = model(X_arbol, y_arbol, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

print("-------------------------------------------------------------------")
huevoread = model(X_huevo,y_huevo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
interrogacionread = model(X_interrogacion,y_interrogacion, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

print("-------------------------------------------------------------------")
print("se calcula el arbol")

dimensiones = [X_arbol.shape[0], 5, 1]

parametros_tanh1 = gradiente_aprendisaje(X_arbol, y_arbol, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Arbol Reading...")
my_image = "test.dibujo.jpg"  
fname = "dibujo/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_arbol = predict4(arbolread["w"], arbolread["b"], my_image)

print("-------------------------------------------------------------------")

print("-------------------------------------------------------------------")
print("se calcula el huevo")



dimensiones = [X_huevo.shape[0], 5, 1]

parametros_tanh2 = gradiente_aprendisaje(X_huevo, y_huevo, dimensiones, theta="tanh",
                   initialization_method="he")

print("Huevo Reading...")
my_predicted_huevo = predict4(huevoread["w"], huevoread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula el interrogacion")

dimensiones = [X_interrogacion.shape[0], 5, 1]

parametros_tanh3 = gradiente_aprendisaje(X_interrogacion, y_interrogacion, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("interrogacion Reading...")
my_predicted_interrogacion = predict4(interrogacionread["w"], interrogacionread["b"], my_image)

print("-------------------------------------------------------------------")
print("arbol")
arbol_data=accuracy(X_arbol, parametros_tanh1,y_arbol, "tanh")
test_data_test_arbol=accuracy(X_test, parametros_tanh1,y_test, "tanh")
print ("huevo")
huevo_data_huevo=accuracy(X_huevo, parametros_tanh2, y_huevo, "tanh")
test_data_huevo_test=accuracy(X_test, parametros_tanh2,y_test, "tanh")
print("test de interrogacion")
interrogacion_data_interrogacion=accuracy(X_interrogacion, parametros_tanh3, y_interrogacion, "tanh")
test_data_triste_test=accuracy(X_test, parametros_tanh3,y_test, "tanh")
print("-------------------------------------------------------------------")
my_predicted_arbol=float(my_predicted_arbol)
my_predicted_huevo=float(my_predicted_huevo)
my_predicted_interrogacion=float(my_predicted_interrogacion)
if(my_predicted_arbol > my_predicted_huevo) and (my_predicted_arbol > my_predicted_interrogacion):
    largest=my_predicted_arbol
    categoria="arbol"
    print("arbol es mayor")
    print (my_predicted_arbol)
    print(arbol_data)
    print(test_data_test_arbol)
    accuracy=arbol_data
    accuracy=float(accuracy)
    text_file = open("bestcase/modulo2.txt", "w")
    text_file.write(categoria)
    text_file = open("bestcase/modulo2predicion.txt", "w")
    text_file.write(str(my_predicted_arbol))
    text_file = open("bestcase/modulo2accuracy.txt", "w")
    text_file.write(str(arbol_data))
    text_file = open("bestcase/accuracy.txt", "w")
    text_file.write(str(accuracy))
elif (my_predicted_huevo > my_predicted_arbol) and (my_predicted_huevo > my_predicted_interrogacion):
    largest=my_predicted_huevo
    categoria="huevo"
    print("huevo es mayor")
    print (my_predicted_huevo)
    print(huevo_data_huevo)
    print(test_data_huevo_test)
    accuracy=huevo_data_huevo
    accuracy=float(accuracy)
    text_file = open("bestcase/modulo2.txt", "w")
    text_file.write(categoria)
    text_file = open("bestcase/modulo2predicion.txt", "w")
    text_file.write(str(my_predicted_huevo))
    text_file = open("bestcase/modulo2accuracy.txt", "w")
    text_file.write(str(huevo_data_huevo))
    text_file = open("bestcase/accuracy.txt", "w")
    text_file.write(str(accuracy))
else:
    largest=my_predicted_interrogacion
    categoria="interrogacion" 
    print("interrogacion es mayor")
    print("Prediccion")
    print (my_predicted_interrogacion)
    print("Accuracy")
    print(interrogacion_data_interrogacion)
    print(test_data_triste_test)
    accuracy=interrogacion_data_interrogacion
    accuracy=float(accuracy)
    text_file = open("bestcase/modulo2.txt", "w")
    text_file.write(categoria)
    text_file = open("bestcase/modulo2predicion.txt", "w")
    text_file.write(str(my_predicted_interrogacion))
    text_file = open("bestcase/modulo2accuracy.txt", "w")
    text_file.write(str(interrogacion_data_interrogacion))
    text_file = open("bestcase/accuracy.txt", "w")
    text_file.write(str(accuracy))

print("-------------------------------------------------------------------")
