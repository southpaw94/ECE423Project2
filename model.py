from sympy import symbols, sin, cos, pi, Matrix, pprint, simplify, diff
from sympy.printing.theanocode import theano_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# DH table of this planar robot given by
# theta_i | d_i | alpha_i | a_i
# --------|-----|---------|----
# theta1  | 0   | pi / 2  | 1
# 0       | d2  | -pi / 2 | 0
# theta3  | 0   | 0       | 1

# define the symbols to be used
theta, d, alpha, a = symbols('theta d alpha a')
theta1, d2, theta3 = symbols('theta1 d2 theta3')
t = symbols('t')

# define the q vector from the DH table
q = Matrix([[theta1, d2, theta3]]).T
print('\nThe q vector describing free parameters is given by:')
pprint(q)

# define standard A matrix
A = Matrix([[cos(theta), -sin(theta), 0, 0],
            [sin(theta), cos(theta), 0, 0],
            [0, 0, 1, d], 
            [0, 0, 0, 1]]) * \
    Matrix([[1, 0, 0, a],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1]])

# find the A matrices described by each row of the DH table
A1 = A.subs([(theta, theta1), (d, 0), (alpha, pi / 2), (a, 1)])
A2 = A.subs([(theta, 0), (d, d2), (alpha, -pi / 2), (a, 0)])
A3 = A.subs([(theta, theta3), (d, 0), (alpha, 0), (a, 1)])

# find final frame WRT base
A30 = A1 * A2 * A3

# Hence, p30 is rows one and two of the final column
# as this is a planar robot with no movement in z-direction
p30 = A30[:2, 3]
print('\nThe position vector WRT base is given by:')
pprint(p30)

# compute the jacobian
J = simplify(p30.jacobian(q))
print('\nHence, the Jacobian matrix is given by:')
pprint(J)

# print the parametric equations describing the circle
print('\nThe parametric equations describing the desired circle are given by:')
print('x(t)=0.6cos(0.5t)+1.2')
print('y(t)=0.6sin(0.5t)+1.0')
p = Matrix([[0.6*cos(0.5*t)+1.2, 0.6*sin(0.5*t)+1.0]]).T

# print the derivatives of the parametric equations
v = diff(p, t)
print('\nHence, the velocity vector is given by:')
pprint(v)

J_inv = J.T * (J * J.T) ** -1
q_dot = J_inv * v

q_start = Matrix([[pi / 2, 0.8, -pi / 2]]).T
dt = 4 * np.pi / 999

#position = np.array(p30.subs([(theta1, q_start[0]), (d2, q_start[1]), (theta3, q_start[2])]))

times = np.linspace(0, 4 * np.pi, 1000)
position = np.zeros((2, len(times)))
position[:, 0] = np.array(p30.subs([(theta1, q_start[0]), (d2, q_start[1]), (theta3, q_start[2])]))[:,0]

fig, ax = plt.subplots()
line, = ax.plot(position[0, :], position[1, :])
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])

# Define expressions for computing the q_dot and p30 matrices
dq = q_dot.subs([(theta1, theta1), (d2, d2), (theta3, theta3), (t, t)])
p = p30.subs([(theta1, theta1), (d2, d2), (theta3, theta3)])

# Complile these expressions into C code for huge performance increase
dq_calc = theano_function([theta1, d2, theta3, t], [dq], allow_input_downcast=True)
p_calc = theano_function([theta1, d2, theta3], [p], allow_input_downcast=True)

for i, time in enumerate(times):
    print(i + 1, '/1000 points calculated    \r', end='', sep='')
    dq = dq_calc(float(q_start[0]), float(q_start[1]), float(q_start[2]), time)
    q_start = q_start + dt * dq
    position[:, i] = np.array(p_calc(float(q_start[0]),
                                     float(q_start[1]),
                                     float(q_start[2])))[:, 0]

def animate(time):
    global position
    #line.set_xdata(position[0, :time])
    #line.set_ydata(position[1, :time])
    line.set_data(position[0, :time], position[1, :time])
    return line,

def init():
    line.set_data([], [])
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(position.shape[1]), interval=5, init_func=init, blit=True, repeat=False)
plt.title('Trajectory of end point')
plt.show()
