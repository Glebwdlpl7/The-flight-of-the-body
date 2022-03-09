import math
import os
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy.optimize import fsolve
from sympy import lambdify

import folium

# Условие
alpha = (math.pi / 180) * float(input("Введите начальный угол: "))
teta = (math.pi / 180) * float(input("Введите азимутальный угол (0 градусов - на север): "))
Mountain_location = [27.988056,86.925278]
H = 8848
g = 9.8
beta = 0.1         #F = -beta*v
v0 = 200
m = 1
dt = 0.001*v0
'''
# Без трения
xs, ys, x1s, y1s = [], [], [], []
x, y, x1, y1, t = 0, 0, 0, 0, 0
vx = v0 * sy.cos(alpha)
vy = v0 * sy.sin(alpha)

while y >= 0:
    t += dt
    vy -= g * dt
    x += vx * dt
    y += vy * dt
    x1 = v0 * sy.cos(alpha) * t
    y1 = v0 * sy.sin(alpha) * t - g * t ** 2 / 2
    x1s.append(x1)
    y1s.append(y1)
    xs.append(x)
    ys.append(y)


# С трением численно
xs2, ys2 = [], []
x, y = 0, 0
vx = v0 * sy.cos(alpha)
vy = v0 * sy.sin(alpha)

while y >= 0:
    xs2.append(x)
    ys2.append(y)
    vx -= (beta * vx / m) * dt
    vy -= (beta * vy / m) * dt + g * dt
    x += vx * dt
    y += vy * dt


'''
# Аналитически с трением
T, A = sy.Symbol('T', nonnegative=True, real=True), sy.Symbol('A', nonnegative=True, real=True) #Время и угол
VX, VY, X, Y = sy.Function('VX'), sy.Function('VY'), sy.Function('X'), sy.Function('Y')

VX = sy.dsolve(VX(T).diff(T) + beta * VX(T) / m, VX(T), ics={VX(0): v0 * sy.cos(A)}).rhs
VY = sy.dsolve(VY(T).diff(T) + g + beta * VY(T) / m, VY(T), ics={VY(0): v0 * sy.sin(A)}).rhs

# ищем Х и Y
С1 = sy.Symbol('С1')
X = sy.integrate(VX, T) + С1
CONST = (sy.solve(X.subs(T, 0), С1))[0]
X = X.subs(С1, CONST)

C2 = sy.Symbol('C2')
Y = sy.integrate(VY, T) + C2
CONST = sy.solve(Y.subs(T, 0) - H, C2)[0]
Y = Y.subs(C2, CONST)

# ищем Tmax
Y_func = lambdify(T, Y.subs(A, alpha), "math")
Tmax = fsolve(Y_func, float(2*v0*sy.sin(alpha)/g))
s = X.subs([(A, alpha), (T, Tmax[0])])
dphi = s/111.1

map = folium.Map(location=[Mountain_location[0], Mountain_location[1]], zoom_start = 8,  tiles="Stamen Terrain")
for coordinates in [[Mountain_location[0],Mountain_location[1]],[Mountain_location[0]+dphi*sy.cos(teta)*sy.cos(abs(Mountain_location[0]*sy.pi/180)),Mountain_location[1]+dphi*sy.sin(teta)*sy.cos(abs(Mountain_location[0]*sy.pi/180))]]:
    folium.Marker(location=coordinates, icon=folium.Icon(color = 'black')).add_to(map)

map.save("map.html")
webbrowser.open('file://' + os.path.realpath("map.html"))
'''
# Ищем оптимальный угол вылета
A_list = np.linspace(0, (math.pi / 2), 100)
X_max = 0
A_max = 0
for a in A_list:
    Y_func = lambdify(T, Y.subs(A, a), "math")
    T_max = fsolve(Y_func, float(2 * v0 * sy.sin(a) / g))
    X_ = X.subs([(A, a), (T, T_max[0])])
    if X_ > X_max:
        X_max = X_
        A_max = a
        T0 = T_max
print('Оптимальный угол:', round(A_max*180 / np.pi, 1), 'градусов\nМаксимальное расстояние: ', round(X_max, 1))


#Рисуем

t1 = np.linspace(0, Tmax, int(Tmax*100//1))
t4 = np.linspace(0, T0, int(T0*100//1))
x_l = [X.subs([(A, alpha), (T, t1[i][0])]) for i in range(len(t1))]
y_l = [Y.subs([(A, alpha), (T, t1[i][0])]) for i in range(len(t1))]
x_max = [X.subs([(A, A_max), (T, t4[j][0])]) for j in range(len(t4))]
y_max = [Y.subs([(A, A_max), (T, t4[j][0])]) for j in range(len(t4))]
plt.plot(xs, ys, 'red', label="Численно без трения")
plt.plot(x1s, y1s, 'blue', label="Аналитически без трения")
plt.plot(xs2, ys2, 'green', label="Численно с трением")
plt.plot(x_l, y_l, 'orange', label="Аналитически с трением")
plt.plot(x_max, y_max, 'black', label="Максимально далекий полет с трением")
plt.title('Траектория')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.grid()
plt.show()
'''

