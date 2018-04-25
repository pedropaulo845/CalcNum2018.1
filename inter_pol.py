import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp

#supondo um conjunto de pontos (x,y) dados por dois arrays, cria a matriz de vandermonde de um polinômio de grau @deg
def vandermonde(x,y,deg):
    A = np.ones((len(x),1))
    for i in range(1,deg+1):
        new_col = np.power(x,i,dtype=np.float64)
        A = np.column_stack([A,new_col])
    return A,y.T

#inserir os arrays dos pontos
x = np.array([-0.5, -0.271, 0., 0.311, 0.5], dtype=np.float64)
y = np.array([0., -1.031, -2., -0.998, 0.], dtype=np.float64)

#supondo fitting de um pol. de grau = n-1 para n pontos (x,y)
deg = len(x)-1

A,b = vandermonde(x,y,deg)  

#coeficientes a_0,a_1,a_2,a_3,a_4... de um polinômio da forma y(x) = a_0 + a_1*x + a_2*x^2 + a+_3*x^3 + ...
#formato do vetor: [a_0, a_1, a_2, ..., a_n]
coefs = np.linalg.solve(A,b)

#cria uma discretização do eixo x
ls = np.linspace(-3,3,1000)

#calcula os valores dos y correspondentes a cada x
f = [coefs[i]*np.power(ls,i,dtype=np.float64) for i in range(deg+1)]
f = sum(f)

plt.plot(x,y,"ro",ls,f,"b")
plt.axis([-3,3,-3,3])
plt.show()
