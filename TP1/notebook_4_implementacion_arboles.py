import numpy as np
import pandas as pd
import math

from collections import Counter
def construir_arbol(instancias, etiquetas, profundidad, criterio, diccColumnas):
	# EJERCICIO EXTRA
	# Agregamos dos parametros, profundida y criterio, para que sea como el arbol de sklearn
	# profundidad se le especifica una profundida maxima y para el algoritmo al llegar a cero
	# por default es -1 para que no pare nunca con este criterio

    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    print(profundidad)
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterio, diccColumnas)
    diccColumnas[pregunta.atributo] = []
    
    # Criterio de corte: ¿Hay ganancia?
    if ganancia == 0 or profundidad == 0:
        #  Si no hay ganancia en separar, no separamos. 
        return Hoja(etiquetas)
    else: 
        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen   , etiquetas_cumplen   , profundidad-1, criterio, diccColumnas)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, profundidad-1, criterio, diccColumnas)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
        
        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)

# Definición de la estructura del árbol. 

class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas))


class Nodo_De_Decision:
    # Un Nodo de Decisión contiene preguntas y una referencia al sub-árbol izquierdo y al sub-árbol derecho
     
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho
        
        
# Definición de la clase "Pregunta"
class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor
    
    def cumple(self, instancia):
        # Devuelve verdadero si la instancia cumple con la pregunta
        return instancia[self.atributo] > self.valor
    
    def __repr__(self):
        return "¿Es el valor para {} mayor a {}?".format(self.atributo, self.valor)

def gini(diccProps):
    impureza = 1
    for elem in diccProps:
        impureza -= diccProps[elem]**2
    return impureza

def entropia(diccProps):
    entropy = 0
    for elem in diccProps:
        entropy += -(diccProps[elem]*math.log(diccProps[elem]))
    return entropy

def criterio(crit,etiquetas):
	criterios = {"gini":gini,"entropy":entropia}
	diccCantidad = Counter(etiquetas)
	diccProps = {}
	for elem in diccCantidad:
		diccProps[elem] = diccCantidad[elem]/len(etiquetas)
	return criterios[crit](diccProps)

def calculoGain(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha, funcionDeGanancia):
    calculo_izq = criterio(funcionDeGanancia, etiquetas_rama_izquierda)
    calculo_der = criterio(funcionDeGanancia, etiquetas_rama_derecha)
    
    total_etiquetas = len(etiquetas_rama_izquierda) + len(etiquetas_rama_derecha)
    
    ganancia = calculo_izq*len(etiquetas_rama_izquierda)/total_etiquetas + calculo_der*len(etiquetas_rama_derecha)/total_etiquetas
    
    return criterio(funcionDeGanancia,etiquetas_rama_izquierda+etiquetas_rama_derecha)-ganancia


def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    # COMPLETAR (recomendamos utilizar máscaras para este punto)

    inst = instancias.copy()
    inst['etiquetas'] = etiquetas
    
    
    instancias_cumplen = inst[pregunta.cumple(inst)]
    instancias_no_cumplen = inst[~pregunta.cumple(inst)]
    
    
    etiquetas_cumplen = instancias_cumplen['etiquetas'].tolist()
    etiquetas_no_cumplen = instancias_no_cumplen['etiquetas'].tolist()
    
    instancias_cumplen.drop('etiquetas', axis=1, inplace=True)
    instancias_no_cumplen.drop('etiquetas', axis=1, inplace=True)

    
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen

def encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterio, diccColumnas):
    max_ganancia = 0
    mejor_pregunta = None
    for columna in instancias.columns:
        listaValores = diccColumnas[columna]
        for valor in listaValores:
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
            ganancia = calculoGain(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha, criterio)
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta
    return max_ganancia, mejor_pregunta

def valoresDondeCambiaEtiqueta(valores, etiquetas):
    tuplasValorEtiqueta = zip(valores, etiquetas)
    res = []
    tuplasOrdenadas = sorted(tuplasValorEtiqueta)
    for i in range(len(tuplasOrdenadas)-1):
        if tuplasOrdenadas[i][1] != tuplasOrdenadas[i+1][1]:
            res.append((tuplasOrdenadas[i][0]+tuplasOrdenadas[i+1][0])/2)
    return res

def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        print()
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")

def predecir(arbol, x_t):
    if isinstance(arbol, Hoja):
        return max(arbol.cuentas, key=arbol.cuentas.get)
    if x_t[arbol.pregunta.atributo] > arbol.pregunta.valor:
        return predecir(arbol.sub_arbol_izquierdo, x_t)
    else:
        return predecir(arbol.sub_arbol_derecho, x_t)
        4


class MiClasificadorArbol(): 
    def __init__(self, columnas, profundidad, criterio):
        self.arbol = None
        self.columnas = columnas
        self.profundidad = profundidad
        self.criterio = criterio
    
    def fit(self, X_train, y_train):
        diccColumnas = {}
        df = pd.DataFrame(X_train, columns=self.columnas)
        for c in self.columnas:
            diccColumnas[c] = valoresDondeCambiaEtiqueta(df[c].tolist(), y_train)
        self.arbol = construir_arbol(df, y_train, self.profundidad, self.criterio, diccColumnas)
        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir(self.arbol, x_t_df) 
            predictions.append(prediction)
        return predictions
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy