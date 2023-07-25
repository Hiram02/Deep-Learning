import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles


# -----------------------------------
#      Parametros de Entrada
# -----------------------------------


# Neuronas capa de entrada
NeuronasEntrada = 2
# Neuronas capa oculta
NeuronasOcultas = 2
# Neuronas capa de salida
NeuronasSalida = 1
# Parametros
alpha = 0.5
Mu = 0
epocaMaxima = 100000
ciclo = 0
errorTotal = 10
errorPermitido = 0.5


# -----------------------------------
#         Bancos de datos
# -----------------------------------


# Funcion senoidal
def make_sin(n_pts, noise=0.3):
    X = np.linspace(0, 1,n_pts)
    Y = np.sin(2 * np.pi * 3 * X)
    Y = Y + np.random.randn(n_pts) * noise
    return  X, Y


# Compuerta XOR
def make_xor(n_pts, noise=0):
    x1 = np.random.randn(int(n_pts / 4), 2) * noise + [0, 1]
    x2 = np.random.randn(int(n_pts / 4), 2) * noise + [1, 0]
    x3 = np.random.randn(int(n_pts / 4), 2) * noise + [0, 0]
    x4 = np.random.randn(int(n_pts / 4), 2) * noise + [1, 1]
    X = np.r_[x1, x2, x3, x4]
    Y = np.r_[np.ones(int(n_pts/2)).T, np.zeros(int(n_pts/2)).T]
    return X, Y


# XOR
data, clases = make_xor(256,0.13)


# Medias lunas
#data, clases = make_moons(n_samples=256, shuffle=False, noise=0.4)


# Circulos concentricos
#data,clases = make_circles(n_samples=256, shuffle=False, noise=0.27)


#Funcion senoidal
#data, clases = make_sin(256,0.45)

#-----------------------------------------------------------------


# separar conjunto de entrenamiento y prueba
data_train, data_test, clases_train, clases_test = train_test_split(data, clases, test_size=0.3, random_state=42)

#---------------------------------------------
#        Conjunto de entrenamiento
#---------------------------------------------

x = data_train
clase = clases_train

filas = len(x)
x0 =[]
for i in range(filas):
    x0.append(x[i][0])

x1 = []
for i in range(filas):
    x1.append(x[i][1])

t = []
for i in range(filas):
    t.append(clase[i])

x = [x0,x1]


#---------------------------------------------
#        Conjunto de prueba
#---------------------------------------------


x0_test =[]
for i in range(len(data_test)):
    x0_test.append(data_test[i][0])

x1_test = []
for i in range(len(data_test)):
    x1_test.append(data_test[i][1])

t_test = []
for i in range(len(data_test)):
    t_test.append(clases_test[i])

x_test = [x0_test, x1_test]

#-----------------------------------------------------


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-5*x))


# Inicializar pesos y bias capa escondida
V = np.zeros((NeuronasEntrada, NeuronasOcultas), dtype=np.float64)
for i in range(NeuronasEntrada):
    for j in range(NeuronasOcultas):
        if i == 0:
            #V[i][j] = np.random.uniform(-0.4, 0.4)
            V[i][j] = np.random.uniform(-0.8, -0.6) #XOR
        else:
            #V[i][j] = np.random.uniform(0.3, -0.4)
            V[i][j] = np.random.uniform(0.7, 0.95) #XOR
V_orig = V #Pesos capa escondida iniciales


# Matriz para guardar (delta-1) para momentum capa escondida
ch = np.zeros((NeuronasEntrada, NeuronasOcultas), dtype=np.float64)


# Inicializar Bias capa escondida
Bv = []
for i in range(NeuronasOcultas):
    #Bv.append(np.random.uniform(-0.3,0.3))
    Bv.append(np.random.uniform(-0.5, 0.5)) #XOR
Bv_orig = Bv # Bias capa escondida inicial


# Guardar (delta-1) de bias para momentum capa escondida
cbh = []
for i in range(len(Bv)):
    cbh.append(0)


# Inicializar pesos y bias capa de salida
W = np.zeros((NeuronasOcultas, NeuronasSalida), dtype=np.float64)
for i in range(NeuronasOcultas):
    for j in range(NeuronasSalida):
        W[i][j] = np.random.uniform(-0.7, 0.7)
        #W[i][j] = np.random.uniform(-0.5,0.5)  #XOR
W_orig = W


# Matriz para guardar (delta-1) para momentum capa salida
co = np.zeros((NeuronasOcultas, NeuronasSalida), dtype=np.float64)


Bw = []
for i in range(NeuronasSalida):
    Bw.append(np.random.uniform(-0.7,0.7))
    #Bw.append(np.random.uniform(-0.5, 0.5)) #XOR
Bw_orig = Bw


# Guardar (delta-1) de bias para momentum capa salida
cbo = []
for i in range(len(Bw)):
    cbo.append(0)


Zin = []
Z = []
for i in range(NeuronasOcultas):
    Zin.append(0)
    Z.append(0)

Yin = []
Y = []
for i in range(NeuronasSalida):
    Yin.append(0)
    Y.append(0)

deltaV = np.zeros((NeuronasEntrada, NeuronasOcultas), dtype=np.float64)
deltaW = np.zeros((NeuronasOcultas, NeuronasSalida), dtype=np.float64)

deltinhaW = []
deltaBw = []
for i in range(NeuronasSalida):
    deltinhaW.append(0)
    deltaBw.append(0)

deltinhaV = []
deltaBv = []
for i in range(NeuronasOcultas):
    deltaBv.append(0)
    deltinhaV.append(0)

matriz = [[],[]]

while (errorTotal > errorPermitido and ciclo < epocaMaxima):
    errorTotal = 0
    ciclo = ciclo +1

    # Fase forward

    # Calculo de la salida de las neuronas de la capa escondida
    for i in range(len(x[0])):      # Recorrer elementos a la entrada
        for j in range(NeuronasOcultas):
            buffer = 0
            for k in range(NeuronasEntrada):
                buffer = buffer + x[k][i] * V[k][j]

            # Calculo de la salida de las neuronas de la capa escondida
            Zin[j] = buffer + Bv[j]
            Z[j] = sigmoid(Zin[j])

        # Calculo de la salida Y, salida de la red
        buffer = 0
        for j in range(NeuronasOcultas):
            buffer = buffer + Z[j] * W[j]
        Yin = buffer + Bw[0]
        Y = sigmoid(Yin)

        # Fase de retropropagacion del error
        # De la salida a la capa escondida
        for j in range(NeuronasSalida):
            deltinhaW[j] = (t[i] - Y[j]) * (Y[j] * (1-Y[j]))

        for j in range(NeuronasOcultas):
            for k in range(NeuronasSalida):
                deltaW[j][k] = alpha*deltinhaW[k]*Z[j]

        for j in range(NeuronasSalida):
            deltaBw[j] = alpha*deltinhaW[j]

        # De la capa escondida a la capa de entrada
        for j in range(NeuronasOcultas):
            for k in range(NeuronasSalida):
                deltinhaV[j] = deltinhaW[k] * W[j][k] * (Z[j] * (1 - Z[j]))

        for j in range(NeuronasEntrada):
            for k in range(NeuronasOcultas):
                deltaV[j][k] = alpha * deltinhaV[k] * x[j][i]
        
        for j in range(NeuronasOcultas):
            deltaBv[j] = alpha * deltinhaV[j]

        # Actualizacion de pesos                      ************MOMENTUM***********
        # De la capa de salida

        # Actualizar pesos
        for j in range(NeuronasOcultas):
            for k in range(NeuronasSalida):
                W[j][k] = W[j][k] + deltaW[j][k] + (Mu * co[j][k])
                co[j][k] = deltaW[j][k] / alpha

        # Actualizar Bias
        for j in range(NeuronasSalida):
            Bw[j] = Bw[j] + deltaBw[j] + (Mu * cbo[j])
            cbo[j] = deltaBw[j] / alpha


        # Capa escondida
        # Actualizar pesos
        for k in range(NeuronasEntrada):
            for j in range(NeuronasOcultas):
                V[k][j] = V[k][j] + deltaV[k][j] + (Mu * ch[k][j])
                ch[k][j] = deltaV[k][j] / alpha

        # Actualizar Bias
        for j in range(NeuronasOcultas):
            Bv[j] = Bv[j] + deltaBv[j] + (Mu * cbh[j])
            cbh[j] = deltaBv[j] / alpha
        
        # Calculo de error total
        for j in range(NeuronasSalida):
            errorTotal = errorTotal + 0.5*((t[i] - Y[j]) ** 2)

    print("Error cuadratico total: {}\nEpocas: {}\n".format(errorTotal,ciclo))
    matriz[0].append(ciclo)
    matriz[1].append(errorTotal)


print("\n")


def prueba(V, Bv, W, Bw, Z, Zin, Y, Yin, x, x0, x1, t, NeuronasOcultas, NeuronasEntrada, NeuronasSalida):
    Yout = []
    for entrada in range(len(x0)):

        for i in range(NeuronasOcultas):
            buffer = 0
            for j in range(NeuronasEntrada):
                buffer = buffer + x[j][entrada] * V[j][i]
            Zin[i] = buffer + Bv[i]
            Z[i] = sigmoid(Zin[i])

        for i in range(NeuronasSalida):
            buffer = 0
            for j in range(NeuronasOcultas):
                buffer = buffer + Z[j] * W[j][i]
            Yin[i] = buffer + Bw[i]
            Y[i] = sigmoid(Yin[i])
            Yout.append(Y[0])
        print("{}    {}    {}   {}".format(str(x0[entrada])[0:5],str(x1[entrada])[0:5],t[entrada],Y[0]))
    return t,Yout


def pecision(clases_reales,clases_predichas):
    aciertos = 0
    clases_asignadas = []
    n_patrones = len(clases_reales)
    for i in range(len(clases_reales)):
        clase_real = clases_reales[i]
        clase_predicha = clases_predichas[i]
        if clase_predicha < 0.5:
            clase_asignada = 0
        elif clase_predicha > 0.5:
            clase_asignada = 1
        else:
            clase_asignada = -1
        clases_asignadas.append(clase_asignada)
        if clase_asignada == clase_real:
            aciertos = aciertos + 1
        else:
            aciertos = aciertos
    precision = (aciertos/n_patrones) * 100
    print("Número de patrones de prueba = ",n_patrones)
    print("Aciertos = ", aciertos)
    print("Precision = ",str(precision)[0:7], "%")
    return precision, clases_asignadas, aciertos


#---------------------------------------
#         Graficas
#---------------------------------------


def graficar_error(matriz):
    plt.plot(matriz[0], matriz[1], 'r')
    plt.axis([0, 1.3 * max(matriz[0]), min(matriz[1]), max(matriz[1])])
    plt.xlim(0,ciclo)
    plt.show()


def ecuaciones_frontera(NeuronasOcultas, V, Bv):
    x = np.arange(-2, 2.5, 0.1)
    fronteras = []
    ecuaciones = []
    for i in range(NeuronasOcultas):
        y = ((-V[0][i] / V[1][i]) * x) - (Bv[i] / V[1][i])
        ecuacion = str((-V[0][i] / V[1][i]))[0:6] + 'x' + ' ' + '-' + ' ' + str(Bv[i] / V[1][i])[0:6]
        fronteras.append(y)
        ecuaciones.append(ecuacion)
        nombre = "ecuacion " + str(i) + "  " + str(ecuacion)
        #print("Las ecuaciones frontera son:")
        #print('\n')
        #print(nombre)
    return x, fronteras, ecuaciones


def graficar(x0,x1,V,Bv,NeuronasOcultas):
    # Datos de entrenamiento
    x_entrenamiento = x0
    y_entrenamiento = x1

    #Datos de prueba
    x_prueba = x0_test
    y_prueba = x1_test


    # Grafica
    afig, ax = plt.subplots()
    # Puntos
    ax.plot(x_entrenamiento, y_entrenamiento, 'o', label='Entrenamiento', color='darkblue', alpha=0.5)
    ax.plot(x_prueba, y_prueba, 'o', label='Prueba', color='seagreen', alpha=0.5)
    # Fronteras
    x, y, ecuacion = ecuaciones_frontera(NeuronasOcultas, V, Bv)
    for i in range(NeuronasOcultas):
        ax.plot(x, y[i], label=ecuacion[i], color='green')
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    #plt.ylim((-1.2,1.8))
    plt.ylim((-0.8,1.8))
    #plt.xlim((-2.8,2.7))
    plt.ylim((-1.5, 1.4)) #circulos
    plt.show()



#---------------------------------------
clases_reales, clases_predichas = \
    prueba(V, Bv, W, Bw, Z, Zin, Y, Yin, x_test, x0_test, x1_test, t_test, NeuronasOcultas, NeuronasEntrada, NeuronasSalida)
#pecision(c_ent_or,c_ent_pre)
pecision(clases_reales,clases_predichas)
ee, frronteras, ewcuaciones = ecuaciones_frontera(NeuronasOcultas, V, Bv)
print("Las ecuaciones frontera son:")
print('\n')
print(ewcuaciones)
graficar_error(matriz)
graficar(x0,x1,V,Bv,NeuronasOcultas)
