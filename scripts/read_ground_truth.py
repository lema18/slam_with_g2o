'exec(%matplotlib inline)'
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
i=int(0)
numero=int(164)
numero2=int(50)
vect_x=np.zeros(shape=(numero,1))
vect_y=np.zeros(shape=(numero,1))
vect_z=np.zeros(shape=(numero,1))
for linea in open('./real.txt'):
    tiempo, x, y, z, q1, q2, q3, q4 = linea.split()
    x=float(x)
    y=float(y)
    z=float(z)
    vect_x[i]=x
    vect_y[i]=y
    vect_z[i]=z
    i=i+1
    if i==numero:
        break
vect_aux_x=np.zeros(shape=(numero2,1))
vect_aux_y=np.zeros(shape=(numero2,1))
vect_aux_z=np.zeros(shape=(numero2,1))
i=0
for linea in open('./odometry.txt'):
    tiempo_2,x_aux, y_aux, z_aux,q1a,q2a,q3a,q4a = linea.split()
    x_aux=float(x_aux)
    y_aux=float(y_aux)
    z_aux=float(z_aux)
    vect_aux_x[i]=x_aux
    vect_aux_y[i]=y_aux
    vect_aux_z[i]=z_aux
    i=i+1
fig=plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(vect_x,vect_y,vect_z,c='r',marker='o')
ax.scatter3D(vect_aux_x,vect_aux_y,vect_aux_z,c='g',marker='^')
plt.show()


