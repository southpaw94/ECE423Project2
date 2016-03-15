from sympy import symbols, sin, cos, pi, Matrix, pprint, simplify, diff
import numpy as np

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
