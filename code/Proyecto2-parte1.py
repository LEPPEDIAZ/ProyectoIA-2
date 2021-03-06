import sys
from matplotlib import pyplot as plt
import numpy as np
sys.path.append("../functions/")
from load_dataset import load_dataset
from load_dataset1 import load_dataset1
from load_dataset2 import load_dataset2
from redesneuronales import (optimizar,propagate2,sigmoid,tanh,relu,feed_forward, feed_act_forward, feed_foward_model,obtener_costo,gradiente_de_sigmoid,gradiente_de_tanh, gradiente_relu, backpropagation,act_backpropagation, modelo_backpropagation, update_parametros, predict4)
import scipy
from PIL import Image
from scipy import ndimage
import skimage
import tkinter 
from tkinter import *

# se importa el data
X_circulo, y_circulo = load_dataset("data/")
index_dibujo = np.argmax(y_circulo); index_circulo = np.argmin(y_circulo)


#se importa data del circulo
X_test, y_test = load_dataset("dibujo/")
index_dibujo = np.argmax(y_test); index_dibujo = np.argmin(y_test)

#se importa cuadrado
X_cuadrado, y_cuadrado = load_dataset1("data/")
index_dibujo = np.argmax(y_cuadrado); index_cuadrado = np.argmin(y_cuadrado)

#se importa el triangulo
X_triangulo, y_triangulo = load_dataset2("data/")
index_dibujo = np.argmax(y_cuadrado); index_triangulo = np.argmin(y_cuadrado)
#resshape origin and examples
circulo_set_x_flatten = X_circulo.reshape(X_circulo.shape[0], -1).T
cuadrado_set_x_flatten = X_cuadrado.reshape(X_cuadrado.shape[0], -1).T
triangulo_set_x_flatten = X_triangulo.reshape(X_triangulo.shape[0], -1).T
test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T
#se tienen 5 capas, input, output y 3 capas escondidas
#constantes que definen el modelo
#se define la estructura de la red neuronal
n_x_circulo = circulo_set_x_flatten.shape[0]
n_x_cuadrado = cuadrado_set_x_flatten.shape[0]
n_x_triangulo = triangulo_set_x_flatten.shape[0]
n_x_test = test_set_x_flatten.shape[0]
num_px=150
n_y = 1 
nn_layers_circulo = [n_x_circulo, 20, 7, 5, n_y]
nn_layers_cuadrado = [n_x_cuadrado, 20, 7, 5, n_y]
nn_layers_triangulo = [n_x_triangulo, 20, 7, 5, n_y]
print("Capas del circulo y cuadrados")
print (nn_layers_circulo)
nn_layers_cuadrado = [n_x_cuadrado, 20, 7, 5, n_y]
print (nn_layers_cuadrado)
print (nn_layers_triangulo)
print(f"""Dimensiones originales para el circulo:\n{20 * '-'}\nData de prueba: {X_cuadrado.shape}, {y_cuadrado.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el cuadrado:\n{20 * '-'}\nData de prueba: {X_circulo.shape}, {y_circulo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el triangulo:\n{20 * '-'}\nData de prueba: {X_triangulo.shape}, {y_triangulo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print ("circulo_set_x_flatten shape: " + str(circulo_set_x_flatten.shape))
print ("cuadrado_set_x_flatten shape: " + str(cuadrado_set_x_flatten.shape))
print ("triangulo_set_x_flatten shape: " + str(triangulo_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

#debido a que se necesita RBG cada pixel es un vector de numeros de 0 a 255.
X_cuadrado = X_cuadrado / 255
X_circulo = X_circulo / 255
X_triangulo = X_triangulo / 255
X_test = X_test / 255
len_cuadrado=len(X_cuadrado)
len_circulo=len(X_circulo)
len_test=len(X_test)
len_triangulo=len(X_triangulo)
print(len_cuadrado)
print(len_circulo)
print(len_test)
print(len_triangulo)
test_dataset=X_circulo, y_circulo
classes = np.array(test_dataset)
index = 25
example = X_circulo[index]
circulo = "circulo"
print ("y = " + str(y_circulo[:, index]) + ", es una foto de un " + circulo)


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
circuloread = model(X_circulo, y_circulo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

print("-------------------------------------------------------------------")
cuadradoread = model(X_cuadrado,y_cuadrado, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
trianguloread = model(X_triangulo,y_triangulo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

print("-------------------------------------------------------------------")
print("se calcula el circulo")

dimensiones = [X_circulo.shape[0], 5, 1]

parametros_tanh1 = gradiente_aprendisaje(X_circulo, y_circulo, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Circulo Reading...")
my_image = "test.dibujo.jpg"  
fname = "dibujo/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_circle = predict4(circuloread["w"], circuloread["b"], my_image)

print("-------------------------------------------------------------------")

print("-------------------------------------------------------------------")
print("se calcula el cuadrado")



dimensiones = [X_cuadrado.shape[0], 5, 1]

parametros_tanh2 = gradiente_aprendisaje(X_cuadrado, y_cuadrado, dimensiones, theta="tanh",
                   initialization_method="he")

print("Cuadrado Reading...")
my_predicted_square = predict4(cuadradoread["w"], cuadradoread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula el triangulo")

dimensiones = [X_triangulo.shape[0], 5, 1]

parametros_tanh3 = gradiente_aprendisaje(X_triangulo, y_triangulo, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Triangulo Reading...")
my_predicted_triangulo = predict4(trianguloread["w"], trianguloread["b"], my_image)

print("-------------------------------------------------------------------")
print("circulo")
circulo_data_circulo=accuracy(X_circulo, parametros_tanh1,y_circulo, "tanh")
test_data_test_circulo=accuracy(X_test, parametros_tanh1,y_test, "tanh")
print ("test de cuadrado")
cuadrado_data_cuadrado=accuracy(X_cuadrado, parametros_tanh2, y_cuadrado, "tanh")
test_data_cuadrado_test=accuracy(X_test, parametros_tanh2,y_test, "tanh")
print("test de triangulo")
triangulo_data_triangulo=accuracy(X_triangulo, parametros_tanh3, y_triangulo, "tanh")
test_data_triangulo_test=accuracy(X_test, parametros_tanh3,y_test, "tanh")
print("-------------------------------------------------------------------")
my_predicted_circle=float(my_predicted_circle)
my_predicted_square=float(my_predicted_square)
my_predicted_triangulo=float(my_predicted_triangulo)
if(my_predicted_circle > my_predicted_triangulo) and (my_predicted_circle > my_predicted_square):
    largest=my_predicted_circle
    categoria="circulo"
    print("circulo es mayor")
    print (my_predicted_circle)
    print(circulo_data_circulo)
    accuracy=circulo_data_circulo
    accuracy=float(accuracy)
    text_file = open("bestcase/modulo1.txt", "w")
    text_file.write(categoria)
    text_file = open("bestcase/modulo1predicion.txt", "w")
    text_file.write(str(my_predicted_circle))
    text_file = open("bestcase/modulo1accuracy.txt", "w")
    text_file.write(str(circulo_data_circulo))
    text_file = open("bestcase/accuracy.txt", "w")
    text_file.write(str(accuracy))
elif (my_predicted_triangulo > my_predicted_circle) and (my_predicted_triangulo > my_predicted_square):
    largest=my_predicted_triangulo
    categoria="triangulo"
    print("triangulo es mayor")
    print (my_predicted_triangulo)
    print(triangulo_data_triangulo)
    accuracy=triangulo_data_triangulo
    accuracy=float(accuracy)
    text_file = open("bestcase/modulo1.txt", "w")
    text_file.write(categoria)
    text_file = open("bestcase/modulo1predicion.txt", "w")
    text_file.write(str(my_predicted_triangulo))
    text_file = open("bestcase/modulo1accuracy.txt", "w")
    text_file.write(str(triangulo_data_triangulo))
    text_file = open("bestcase/accuracy.txt", "w")
    text_file.write(str(accuracy))
else:
    largest=my_predicted_square
    categoria="square" 
    print("square es mayor")
    print("Prediccion")
    print (my_predicted_square)
    print("Accuracy")
    print(square_data_square)
    accuracy=square_data_square
    accuracy=float(accuracy)
    text_file = open("bestcase/modulo1.txt", "w")
    text_file.write(categoria)
    text_file = open("bestcase/modulo1predicion.txt", "w")
    text_file.write(str(my_predicted_square))
    text_file = open("bestcase/modulo1accuracy.txt", "w")
    text_file.write(str(square_data_square))
    text_file = open("bestcase/accuracy.txt", "w")
    text_file.write(str(accuracy))

print("-------------------------------------------------------------------")
