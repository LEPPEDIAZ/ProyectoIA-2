import sys
from matplotlib import pyplot as plt
import numpy as np
sys.path.append("../functions/")
from load_dataset import load_dataset
from load_dataset1 import load_dataset1
from load_dataset2 import load_dataset2
from load_dataset3 import load_dataset3
from load_dataset4 import load_dataset4
from load_dataset5 import load_dataset5
from load_dataset6 import load_dataset6
from load_dataset7 import load_dataset7
from load_dataset8 import load_dataset8
from redesneuronales import (initeparametros,optimizar,sigmoid_backward,propagate,propagate2,linear_backward,L_model_forward,initeparametros2,calcular_costo,linear_forward,linear_activation_forward,sigmoid,sigmoid2,tanh,relu,relu2,leaky_relu,feed_forward, feed_act_forward, feed_foward_model,obtener_costo,gradiente_de_sigmoid,gradiente_de_tanh, gradiente_relu, backpropagation,act_backpropagation, modelo_backpropagation, update_parametros, predict, predict2,predict3)
import scipy
from PIL import Image
from scipy import ndimage
import skimage
import tkinter 
from tkinter import *

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

#se importa el triangulo
X_triangulo, y_triangulo = load_dataset2("data2/")
index_dibujo = np.argmax(y_cuadrado); index_triangulo = np.argmin(y_triangulo)
#se importa el arbol
X_arbol, y_arbol = load_dataset3("data2/")
index_dibujo = np.argmax(y_arbol); index_arbol = np.argmin(y_arbol)
#se importa el huevo
X_huevo, y_huevo = load_dataset4("data2/")
index_dibujo = np.argmax(y_huevo); index_huevo = np.argmin(y_huevo)
#se importa la interrogacion
X_interrogacion, y_interrogacion = load_dataset5("data2/")
index_dibujo = np.argmax(y_interrogacion); index_interrogacion = np.argmin(y_interrogacion)
#se importa la casa
X_casa, y_casa = load_dataset6("data2/")
index_dibujo = np.argmax(y_interrogacion); index_casa = np.argmin(y_interrogacion)
#se importa la feliz
X_feliz, y_feliz = load_dataset7("data2/")
index_dibujo = np.argmax(y_interrogacion); index_feliz = np.argmin(y_interrogacion)
#se importa la triste
X_triste, y_triste = load_dataset8("data2/")
index_dibujo = np.argmax(y_interrogacion); index_triste = np.argmin(y_interrogacion)
#resshape origin and examples
circulo_set_x_flatten = X_circulo.reshape(X_circulo.shape[0], -1).T
cuadrado_set_x_flatten = X_cuadrado.reshape(X_cuadrado.shape[0], -1).T
triangulo_set_x_flatten = X_triangulo.reshape(X_triangulo.shape[0], -1).T
arbol_set_x_flatten = X_arbol.reshape(X_arbol.shape[0], -1).T
huevo_set_x_flatten = X_huevo.reshape(X_huevo.shape[0], -1).T
test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T
#se tienen 5 capas, input, output y 3 capas escondidas
#constantes que definen el modelo
#se define la estructura de la red neuronal
n_x_circulo = circulo_set_x_flatten.shape[0]
n_x_cuadrado = cuadrado_set_x_flatten.shape[0]
n_x_triangulo = triangulo_set_x_flatten.shape[0]
n_x_test = test_set_x_flatten.shape[0]
n_x_arbol = arbol_set_x_flatten.shape[0]
n_x_huevo = huevo_set_x_flatten.shape[0]
num_px=150
n_y = 1 
nn_layers_circulo = [n_x_circulo, 20, 7, 5, n_y]
nn_layers_cuadrado = [n_x_cuadrado, 20, 7, 5, n_y]
nn_layers_triangulo = [n_x_triangulo, 20, 7, 5, n_y]
nn_layers_arbol = [n_x_arbol, 20, 7, 5, n_y]
nn_layers_huevo = [n_x_huevo, 20, 7, 5, n_y]
print("Capas del circulo y cuadrados")
print (nn_layers_circulo)
nn_layers_cuadrado = [n_x_cuadrado, 20, 7, 5, n_y]
print (nn_layers_cuadrado)
print (nn_layers_triangulo)
print (nn_layers_arbol)
print (nn_layers_huevo)
print(f"""Dimensiones originales para el circulo:\n{20 * '-'}\nData de prueba: {X_cuadrado.shape}, {y_cuadrado.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el cuadrado:\n{20 * '-'}\nData de prueba: {X_circulo.shape}, {y_circulo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el triangulo:\n{20 * '-'}\nData de prueba: {X_triangulo.shape}, {y_triangulo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el arbol:\n{20 * '-'}\nData de prueba: {X_arbol.shape}, {y_arbol.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el huevo:\n{20 * '-'}\nData de prueba: {X_huevo.shape}, {y_huevo.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el signo de interrogacion:\n{20 * '-'}\nData de prueba: {X_casa.shape}, {y_casa.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el signo de interrogacion:\n{20 * '-'}\nData de prueba: {X_feliz.shape}, {y_feliz.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el signo de interrogacion:\n{20 * '-'}\nData de prueba: {X_triste.shape}, {y_triste.shape}
Test: {X_test.shape}, {y_test.shape}""")
print(f"""Dimensiones originales para el signo de interrogacion:\n{20 * '-'}\nData de prueba: {X_interrogacion.shape}, {y_interrogacion.shape}
Test: {X_test.shape}, {y_test.shape}""")
print ("circulo_set_x_flatten shape: " + str(circulo_set_x_flatten.shape))
print ("cuadrado_set_x_flatten shape: " + str(cuadrado_set_x_flatten.shape))
print ("triangulo_set_x_flatten shape: " + str(triangulo_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("arbol_set_x_flatten shape: " + str(arbol_set_x_flatten.shape))
print ("arbol_set_x_flatten shape: " + str(huevo_set_x_flatten.shape))
#debido a que se necesita RBG cada pixel es un vector de numeros de 0 a 255.
X_cuadrado = X_cuadrado / 255
X_circulo = X_circulo / 255
X_triangulo = X_triangulo / 255
X_test = X_test / 255
X_arbol = X_arbol / 255
X_huevo = X_huevo / 255
X_interrogacion = X_interrogacion / 255
X_casa = X_casa / 255
X_feliz = X_feliz / 255
X_triste = X_triste / 255
len_cuadrado=len(X_cuadrado)
len_circulo=len(X_circulo)
len_test=len(X_test)
len_triangulo=len(X_triangulo)
len_arbol=len(X_arbol)
print(len_cuadrado)
print(len_circulo)
print(len_test)
print(len_triangulo)
print(len_arbol)
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

    return print (f"El accuracy es: {accuracy:.2f}%.")
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
circuloread = model(X_circulo, y_circulo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)

print("-------------------------------------------------------------------")
cuadradoread = model(X_cuadrado,y_cuadrado, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
trianguloread = model(X_triangulo,y_triangulo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
arbolread = model(X_arbol,y_arbol, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
huevoread = model(X_huevo,y_huevo, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
interrogacionread = model(X_interrogacion,y_interrogacion, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
casaread = model(X_casa,y_casa, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
felizread = model(X_feliz,y_feliz, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
tristeread = model(X_triste,y_triste, X_test, y_test, num_iterations = 100, learning_rate = 0.005, print_cost = False)
print("-------------------------------------------------------------------")
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")
print("se calcula el circulo")

dimensiones = [X_circulo.shape[0], 5, 5, 1]

parametros_tanh1 = gradiente_aprendisaje(X_circulo, y_circulo, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Circulo Reading...")
my_image = "test.dibujo.jpg"  
fname = "dibujo/" + my_image
image = np.array(plt.imread(fname))
my_image = skimage.transform.resize(image, output_shape=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_circle = predict3(circuloread["w"], circuloread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")

print("-------------------------------------------------------------------")
print("se calcula el cuadrado")



dimensiones = [X_cuadrado.shape[0], 5, 5, 1]

parametros_tanh2 = gradiente_aprendisaje(X_cuadrado, y_cuadrado, dimensiones, theta="tanh",
                   initialization_method="he")

print("Cuadrado Reading...")
my_predicted_square = predict3(cuadradoread["w"], cuadradoread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula el triangulo")

dimensiones = [X_triangulo.shape[0], 5, 5, 1]

parametros_tanh3 = gradiente_aprendisaje(X_triangulo, y_triangulo, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Triangulo Reading...")
my_predicted_triangulo = predict3(trianguloread["w"], trianguloread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")
print("se calcula el arbol")

dimensiones = [X_arbol.shape[0], 5, 5, 1]

parametros_tanh4 = gradiente_aprendisaje(X_arbol, y_arbol, dimensiones, theta="tanh",
                   initialization_method="he")
print("-------------------------------------------------------------------")
print("Arbol Reading...")
my_predicted_arbol = predict3(arbolread["w"], arbolread["b"], my_image)
#my_predicted_image = predict3(d["w"], d["b"], image)
#print("y = " + str(np.squeeze(my_predicted_image)) + ", el algoritmo predice que es \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\".")
print("-------------------------------------------------------------------")

print("se calcula el huevo")



dimensiones = [X_huevo.shape[0], 5, 5, 1]

parametros_tanh3 = gradiente_aprendisaje(X_huevo, y_huevo, dimensiones, theta="tanh",
                   initialization_method="he")

print("Huevo Reading...")
my_predicted_huevo = predict3(huevoread["w"],huevoread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula el signo de interrogacion")

dimensiones = [X_interrogacion.shape[0], 5, 5, 1]

parametros_tanh4 = gradiente_aprendisaje(X_interrogacion, y_interrogacion, dimensiones, theta="tanh",
                   initialization_method="he")

print("Interrogacion Reading...")
my_predicted_interrogacion = predict3(interrogacionread["w"],interrogacionread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula casa")

dimensiones = [X_casa.shape[0], 5, 5, 1]

parametros_tanh5 = gradiente_aprendisaje(X_casa, y_casa, dimensiones, theta="tanh",
                   initialization_method="he")

print("Casa Reading...")
my_predicted_casa = predict3(casaread["w"],casaread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula la cara feliz")

dimensiones = [X_feliz.shape[0], 5, 5, 1]

parametros_tanh6 = gradiente_aprendisaje(X_feliz, y_feliz, dimensiones, theta="tanh",
                   initialization_method="he")

print("Feliz Reading...")
my_predicted_feliz = predict3(felizread["w"],felizread["b"], my_image)
print("-------------------------------------------------------------------")
print("se calcula la cara triste")

dimensiones = [X_triste.shape[0], 5, 5, 1]

parametros_tanh4 = gradiente_aprendisaje(X_triste, y_triste, dimensiones, theta="tanh",
                   initialization_method="he")

print("Triste Reading...")
my_predicted_triste = predict3(tristeread["w"],tristeread["b"], my_image)
print("-------------------------------------------------------------------")

if ( my_predicted_circle > 0.000001 ):
    #0.0000041409869
   print ("Es un circulo")
   circulo_data=accuracy(X_circulo, parametros_tanh1,y_circulo, "tanh")
   test_data=accuracy(X_test, parametros_tanh1,y_test, "tanh")

else:
   print ("no es un circulo")

cuadrado_data=accuracy(X_cuadrado, parametros_tanh2, y_cuadrado, "tanh")
test_data2=accuracy(X_test, parametros_tanh2,y_test, "tanh")
if ( my_predicted_square > 0.999999):
    #0.99999586
    #0.9999781
    #0.99999172
   top=Tk()
   top.geometry("180x180")
   top.configure(background='blue')
   Lb1 = Listbox(top)
   Lb1.insert(1, "El programa es:")
   Lb1.insert(2, "Un circulo")
   Lb1.insert(3, (print(accuracy(X_cuadrado, parametros_tanh2, y_cuadrado, "tanh"))))
   Lb1.insert(4,  (accuracy(X_test, parametros_tanh2,y_test, "tanh")))
   Lb1.pack()
   top.mainloop()
   
 
else:
   print ("No es un cuadrado")

if (0.99999172 > 0.999999):
    #0.99999586
    #0.9999781
    #0.99999172
   print ("true")
else:
   print ("false")


print("-------------------------------------------------------------------")
