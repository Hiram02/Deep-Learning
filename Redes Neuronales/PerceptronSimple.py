import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from sklearn.model_selection import train_test_split


# -----------------------------------
#         Bancos de datos
# -----------------------------------

def aa():
    v1 = np.array([3,0,3,0,1,1,0,0])
    v2 = np.array([1,0,1,0,3,3,0,0])
    Y = np.r_[0,1]
    X = np.r_[[v1], [v2]]
    return X, Y

def make_or(n_pts, noise=0.3):
    x1 = np.random.randn(int(n_pts / 4), 2) * noise + [0, 0]
    x2 = np.random.randn(int(n_pts / 4), 2) * noise + [0, 4]
    x3 = np.random.randn(int(n_pts / 4), 2) * noise + [4, 0]
    x4 = np.random.randn(int(n_pts / 4), 2) * noise + [4, 4]
    # Datos / patrone
    X = np.r_[x1, x2, x3, x4]
    # clases
    Y = np.r_[np.zeros(int(n_pts / 4)).T, np.ones(int(n_pts / 4 * 3)).T]
    return X, Y


def make_and(n_pts, noise=0.3):
    x1 = np.random.rand(int(n_pts / 4), 2) * noise + [0, 0]
    x2 = np.random.rand(int(n_pts / 4), 2) * noise + [0, 4]
    x3 = np.random.rand(int(n_pts / 4), 2) * noise + [4, 0]
    x4 = np.random.rand(int(n_pts / 4), 2) * noise + [4, 4]
    # Datos / patrones
    X = np.r_[x1, x2, x3, x4]
    # clases
    Y = np.r_[np.zeros(int(n_pts / 4 * 3)).T, np.ones(int(n_pts / 4)).T]
    return X, Y


# -----------------------------------
#           FUNCIONES
# -----------------------------------


def vectores_aumentados(data, aumento):
    filas = len(data)
    columnas = len(data[1]) + 1
    dataset = np.zeros(shape=(filas, columnas))
    for i in range(len(data)):
        patron = data[i]
        patron = np.append([patron], [aumento])
        dataset[i] = patron
    return dataset


def activacion(patron, pesos, theta):
    a = 0
    # todos lo rasgos excepto el aumentado
    for i in range(len(patron) - 1):
        rasgo = patron[i]
        peso = pesos[i]
        a = a + (rasgo * peso)
    a = a + theta
    return a


def grad_error(s, t, patron):
    num_rasgos = len(patron)
    gradientes = []
    for i in range(num_rasgos):
        rasgo = patron[i]
        grad_rasgo = s * (1 - s) * (s - t) * rasgo
        gradientes.append(grad_rasgo)
    return gradientes


def pesos_actualizados(patron, pesos, gradientes, alpha, theta):
    num_rasgos = len(patron)
    deltas = np.zeros(shape=(0, 0))
    for i in range(num_rasgos):
        if i == num_rasgos - 1:
            rasgo = patron[i]
            d_peso = -alpha * gradientes[i] * rasgo
            theta = theta + d_peso
            deltas = np.append(deltas, d_peso)
        else:
            rasgo = patron[i]
            d_peso = -alpha * gradientes[i] * rasgo
            pesos[i] = pesos[i] + d_peso
            deltas = np.append(deltas, d_peso)
    return pesos, theta, deltas


def prueba(data_test, clases_test, pesos, theta):
    data_prueba = vectores_aumentados(data_test, -1)
    aciertos = 0
    clases_asignadas = []
    n_patrones = len(data_prueba)
    for i in range(len(data_prueba)):
        patron = data_prueba[i]
        clase_real = clases_test[i]
        a = activacion(patron, pesos, theta)
        s = pow(1 + pow(e, -a), -1)
        if s < 0.5:
            clase_asignada = -1
        elif s > 0.5:
            clase_asignada = 1
        else:
            clase_asignada = 0
        clases_asignadas.append(clase_asignada)
        if clase_asignada == clase_real:
            aciertos = aciertos + 1
        else:
            aciertos = aciertos
    precision = (aciertos/n_patrones) * 100
    return precision, clases_asignadas, aciertos


# -----------------------------------
#           OPERACIONES
# -----------------------------------


# constante de Euler
e = math.e
# data -> patrones
# clases -> clases o valor deseado
data, clases = aa()
print(data)

x_min = min(data[:, 0])
x_max = max(data[:, 0])
# remmplazar valores = 0 por -1
clases[clases == 0] = -1

# separar conjunto de entrenamiento y prueba
data_train = data
data_test = data
clases_train = clases
clases_test = clases
#data_train, data_test, clases_train, clases_test = train_test_split(data, clases, test_size=0.3, random_state=42)
# vectores aumentados
dataset = vectores_aumentados(data_train, 0.1)
# numero de patrones
num_patrones = len(dataset)

# -----------------------------------
#           Entrenamiento
# -----------------------------------


# inicializar vector de pesos con valores random
cant_rasgos = len(data[0])
#pesos = np.random.rand(1,cant_rasgos)
#pesos = pesos[0,:]
pesos = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
pesos_iniciales = pesos.copy()

# inicializar alpha (tasa de aprendizaje)
#alpha = np.random.normal()
alpha = 0.25
# inicializar theta
#theta = np.random.normal()
theta = 0.1
theta_inicial = theta
# contador de epocas
epocas = 0
# inicial con error del 100%
error_global = 1
errores_globales = []

while error_global > 0 and epocas < 35:
    errores = np.zeros(shape=(0, 0))
    for i in range(len(dataset)):
        patron = dataset[i]
        t = clases_train[i]
        a = activacion(patron, pesos, theta)
        s = pow(1 + pow(e, -a), -1)
        gradientes = grad_error(s, t, patron)
        pesos, theta, deltas = pesos_actualizados(patron, pesos, gradientes, alpha, theta)
        # calcular error individual
        delta = s - t
        error = pow(delta, 2) / 2
        errores = np.append(errores, error)
    epocas = epocas + 1
    # calculo de error
    for error in errores:
        error_global = error_global + error
    error_global = error_global / num_patrones
    errores_globales.append(error_global)

# -----------------------------------
#      Frontera de decision
# -----------------------------------


# valores finales de p1 y p2
p1 = pesos[0]
p2 = pesos[1]
# rango en x para linea divisora
x = np.arange(x_min, x_max, 0.1)
# funcion delimitadora
y = (x * (-p1 / p2)) + (-theta / p2)
ecuacion = str(-p1/p2)[0:7] + 'x' + ' ' + '+' + ' ' + str(-theta/p2)[0:6]


# -----------------------------------
#             Prueba
# -----------------------------------


precision, clases_asignadas, aciertos = prueba(data_test, clases_test, pesos, theta)


# -----------------------------------
#      Imprimir en pantalla
# -----------------------------------

print("pesos iniciales= ",pesos_iniciales)
print("pesos finales=  ", pesos)
print("theta inicial= ",theta_inicial)
print("theta final= ", theta)
print("Alpha= ",alpha)
print("# epocas= ", epocas)
print("Aciertos= ",aciertos)
print("Precisión= ",precision," %")
print("Error= ", error_global)
print("ecuacion= ",ecuacion)


# -----------------------------------
#            Grafica
# -----------------------------------

x_g = np.arange(0, epocas, 1)
plt.plot(x_g,errores_globales)


# DATOS DE ENTRENAMIENTO
x_entrenamiento = data_train[:, 0]
y_entrenamiento = data_train[:, 1]

# DATOS DE PRUEBA
x_prueba = data_test[:, 0]
y_prueba = data_test[:, 1]

afig, ax = plt.subplots()
# conjunto de entrenamiento
ax.plot(x_entrenamiento, y_entrenamiento, 'o', label='Entrenamiento', color='darkblue', alpha=0.5)
# conjunto de prueba
ax.plot(x_prueba, y_prueba, 'o', label='Prueba', color='seagreen', alpha=0.6)
# Frontera de decision
ax.plot(x, y, label=ecuacion, color='gold')
legend = ax.legend(loc='lower left', shadow=True, fontsize='small')
#plt.show()

