#import os
#os.system("python Proyecto2-parte1.py")
#os.system("python Proyecto2-parte2.py")
#os.system("python Proyecto2-parte3.py")
import tkinter
from tkinter import *

#cargando modelo 1
with open ("bestcase/modulo1.txt", "r") as myfile:
    data=myfile.readlines()
    print(data)
x = []
file_in = open('bestcase/modulo1accuracy.txt', 'r')
for accuracymodelo1 in file_in.read().split('\n'):
    if accuracymodelo1.isdigit():
        x.append(float(accuracymodelo1))
accuracymodelo1=float(accuracymodelo1)
print(accuracymodelo1)
a = []
file_in = open('bestcase/modulo1predicion.txt', 'r')
for predicionmodelo1 in file_in.read().split('\n'):
    if predicionmodelo1.isdigit():
        a.append(float(predicionmodelo1))
predicionmodelo1=float(predicionmodelo1)
print(predicionmodelo1)

#cargando modelo 2
with open ("bestcase/modulo2.txt", "r") as myfile:
    data2=myfile.readlines()
    print(data2)
x = []
file_in = open('bestcase/modulo2accuracy.txt', 'r')
for accuracymodelo2 in file_in.read().split('\n'):
    if accuracymodelo2.isdigit():
        x.append(float(accuracymodelo2))
accuracymodelo2=float(accuracymodelo2)
print(accuracymodelo2)
a = []
file_in = open('bestcase/modulo2predicion.txt', 'r')
for predicionmodelo2 in file_in.read().split('\n'):
    if predicionmodelo2.isdigit():
        a.append(float(predicionmodelo2))
predicionmodelo2=float(predicionmodelo2)
print(predicionmodelo2)
#cargando modelo 3
with open ("bestcase/modulo3.txt", "r") as myfile:
    data3=myfile.readlines()
    print(data3)
x = []
file_in = open('bestcase/modulo3accuracy.txt', 'r')
for accuracymodelo3 in file_in.read().split('\n'):
    if accuracymodelo3.isdigit():
        x.append(float(accuracymodelo3))
accuracymodelo3=float(accuracymodelo3)
print(accuracymodelo3)
a = []
file_in = open('bestcase/modulo3predicion.txt', 'r')
for predicionmodelo3 in file_in.read().split('\n'):
    if predicionmodelo3.isdigit():
        a.append(float(predicionmodelo3))
predicionmodelo3=float(predicionmodelo3)
print(predicionmodelo3)

#se inician comparaciones
if(accuracymodelo3 >= accuracymodelo2) and (accuracymodelo3 >= accuracymodelo1) and (predicionmodelo3 > predicionmodelo2) and (predicionmodelo3 > predicionmodelo1):
    print(data3)
    top = Tk()
    top.geometry("200x200")
    top.configure(background='blue')
    Lb1 = Listbox(top)
    Lb1.insert(1, "Creo que es", data3)
    Lb1.insert(2, "con",accuracymodelo3 ,"%")
    Lb1.insert(3, predicionmodelo3 , "%")
    Lb1.pack()
    top.mainloop()
    print("Creo que es",data3, "con", accuracymodelo3 ,"%", " de seguridad")
    print("las siguientes opciones son", data2,"%", accuracymodelo2, "y", data, "%",accuracymodelo1)
if(accuracymodelo2 >= accuracymodelo1) and (accuracymodelo2 >= accuracymodelo3) and (predicionmodelo2 > predicionmodelo3) and (predicionmodelo2 > predicionmodelo1):
    print(data3)
    print("Creo que es",data2, "con", accuracymodelo2 ,"%", " de seguridad")
    print("las siguientes opciones son", data3,"%", accuracymodelo3, "y", data,"%", accuracymodelo1)
if(accuracymodelo1 >= accuracymodelo3) and (accuracymodelo1 >= accuracymodelo2) and (predicionmodelo1 > predicionmodelo3) and (predicionmodelo1 > predicionmodelo2):
    print(data3)
    print("Creo que es",data, "con", accuracymodelo1 ,"%", " de seguridad")
    print("las siguientes opciones son", data3,"%", accuracymodelo3, "y", data2, "%",accuracymodelo2)