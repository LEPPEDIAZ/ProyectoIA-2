import os as os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../functions/")
from load_dataset import load_dataset
from redesneuronales import (initeparametros,sigmoid,tanh,relu,leaky_relu,feed_forward, feed_act_forward, feed_foward_model,obtener_costo,gradiente_de_sigmoid,gradiente_de_tanh, gradiente_relu, backpropagation,act_backpropagation, modelo_backpropagation, update_parametros)


# se importa el data
X_circulo, y_circulo = load_dataset("../data/circulodata")
index_dibujo = np.argmax(y_circulo); index_circulo = np.argmin(y_circulo)
plt.subplot(1, 2, 1)
plt.imshow(X_circulo[:, index_dibujo].reshape(150, 150, 3))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(X_circulo[:, index_circulo].reshape(150, 150, 3))
plt.axis("off");

X_test, y_test = load_dataset("../data/dibujo")
index_dibujo = np.argmax(y_test); index_circulo = np.argmin(y_test)
plt.subplot(1, 2, 1)
plt.imshow(X_test[:, index_dibujo].reshape(150, 150, 3))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(X_test[:, index_circulo].reshape(150, 150, 3))
plt.axis("off");


print(f"""Dimensiones originales:\n{20 * '-'}\nData de prueba: {X_circulo.shape}, {y_circulo.shape}
Test: {X_test.shape}, {y_test.shape}""")


X_circulo = X_circulo / 255
X_test = X_test / 255

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


def gradiente_aprendisaje(
        X, y, dimensiones, curvaaprendisaje=0.01, num_iterations=1000,
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

  
    plt.figure(figsize=(10, 6))
    plt.plot(costo_list)
    plt.xlabel("Iterations (per hundreds)")
    plt.ylabel("Loss")
    plt.title(f"Loss curve for the learning rate = {curvaaprendisaje}")

    return parametros


def accuracy(X, parametros, y,activacion="relu"):
    
    probs, caches = feed_foward_model(X, parametros, activacion)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100

    return print (f"El accuracy es: {accuracy:.2f}%.")


dimensiones = [X_circulo.shape[0], 5, 5, 1]

parametros_tanh = gradiente_aprendisaje(
    X_circulo, y_circulo, dimensiones, curvaaprendisaje=0.03, num_iterations=1000,
    theta="tanh")

accuracy(X_test, parametros_tanh, y_test, activacion="tanh")

parametros_relu = gradiente_aprendisaje(
    X_circulo, y_circulo, dimensiones, curvaaprendisaje=0.03, num_iterations=1000,
    theta="relu")

accuracy(X_test, parametros_relu, y_test, activacion="relu")


parametros_tanh = gradiente_aprendisaje(X_circulo, y_circulo, dimensiones, theta="tanh",
                   initialization_method="he")

accuracy(X_test, parametros_tanh, y_test, "tanh")

parametros_tanh = gradiente_aprendisaje(X_circulo, y_circulo, dimensiones, theta="tanh",
                   initialization_method="xavier")

accuracy(X_test, parametros_tanh, y_test, "tanh")