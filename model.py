from sympy import symbols, sin, cos, pi, Matrix, pprint, simplify, diff, Identity
from sympy.printing.theanocode import theano_function
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib as mpl

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
print('Computing the individual A matrices...')
A1 = A.subs([(theta, theta1), (d, 0), (alpha, pi / 2), (a, 1)])
A2 = A.subs([(theta, 0), (d, d2), (alpha, -pi / 2), (a, 0)])
A3 = A.subs([(theta, theta3), (d, 0), (alpha, 0), (a, 1)])

# find A matrices for second and final frames
print('Computing the A matrices for frames 2 and 3 WRT base...')
A20 = A1 * A2
A30 = A1 * A2 * A3

# Hence, p30 is rows one and two of the final column
# as this is a planar robot with no movement in z-direction
p10 = A1[:2, 3]
p20 = A20[:2, 3]
p30 = A30[:2, 3]

# compute the jacobian
print('Computing the Jacobian matrix...')
J = simplify(p30.jacobian(q))

# print the parametric equations describing the circle
p = Matrix([[0.6*cos(0.5*t)+1.2, 0.6*sin(0.5*t)+1.0]]).T

# print the derivatives of the parametric equations
print('Computing the cartesian velocity vector...')
v = diff(p, t)

# compute the pseudo inverse of the symbolic jacobian matrix
if os.path.isfile('jacobian_inverse.pkl') == True:
    print('Un-pickling the pseudo-inverse Jacobian file')
    J_inv = pickle.load(open('jacobian_inverse.pkl', 'rb'))
else:
    print('Computing the pseudo-inverse Jacobian, this may take some time...')
    J_inv = J.T * (J * J.T) ** -1

# compute the null solution
print('Computing the null solution...')
p_obs = Matrix([[1, 0]]).T
potential = Matrix(np.zeros((3, 1)))
potential[:2, 0] = p20 - Matrix([[1.5, 0]]).T
potential = potential.T * potential
eta = potential.jacobian(q)

q_dot = J_inv * v + (Identity(3).as_explicit() - J_inv * J) * eta.T * -0.9

q_start = Matrix([[pi / 2, 0.8, -pi / 2]]).T
dt = 4 * np.pi / 999

times = np.linspace(0, 4 * np.pi, 1000)
position3 = np.zeros((2, len(times)))
position1 = np.zeros((2, len(times)))
position2 = np.zeros((2, len(times)))

q_vals = np.zeros((3, len(times)))
position3[:, 0] = np.array(p30.subs([(theta1, q_start[0]), (d2, q_start[1]), (theta3, q_start[2])]))[:,0]

fig, ax = plt.subplots()
ax.set_aspect('equal')
line, = ax.plot(position3[0, :], position3[1, :])
floor = plt.Polygon([[-1.2, -0.2], [2, -0.2], [2, -1], [-1.2, -1]], color='lightblue')
wall = plt.Polygon([[-1.2, -0.2], [-1.2, 2], [-1, 2], [-1, -0.2]], color='lightblue')
ax.add_patch(floor)
ax.add_patch(wall)

link1_pos = [[-0.1, 0], [0.1, 0], [0.1, 0.9], [0.2, 0.9], [0.2, 1.1], [-0.2, 1.1], [-0.2, 0.9], [-0.1, 0.9]]
# add the first link of the robot
#link1 = plt.Rectangle([-0.1, 0], 0.2, 1, 45)
link1 = plt.Polygon([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], color='darkgoldenrod')
#link1._transform = tr
ax.add_patch(link1)

# add the second link
link2 = plt.Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], color='cyan')
ax.add_patch(link2)

joint2 = plt.Circle([0, 0], 0.015)
ax.add_patch(joint2)

# add the third joint (revolute)
joint3 = plt.Circle([0, 0], 0.075, color='green')
ax.add_patch(joint3)

line1 = plt.Line2D([0, 0], [0, 0], linewidth=5)

# add the final link
link3 = plt.Polygon([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], color='purple')
ax.add_patch(link3)

# add the base of the robot
circ = plt.Circle([0, 0], 0.1, color='orange')
ax.add_patch(circ)
base = plt.Polygon([[-0.1, 0], [0.1, 0], [0.1,-0.2], [-0.1, -0.2]], color='orange')
ax.add_patch(base)

ax.set_xlim([-1.2, 3])
ax.set_ylim([-1, 2])

# Define expressions for computing the q_dot and p30 matrices
dq = q_dot.subs([(theta1, theta1), (d2, d2), (theta3, theta3), (t, t)])
p3 = p30.subs([(theta1, theta1), (d2, d2), (theta3, theta3)])
p2 = p20.subs([(theta1, theta1), (d2, d2)])
p1 = p10.subs([(theta1, theta1)])

# Complile these expressions into C code for huge performance increase
dq_calc = theano_function([theta1, d2, theta3, t], [dq], allow_input_downcast=True)
p3_calc = theano_function([theta1, d2, theta3], [p3], allow_input_downcast=True)
p2_calc = theano_function([theta1, d2], [p2], allow_input_downcast=True)
p1_calc = theano_function([theta1], [p1], allow_input_downcast=True)

print('Computing the position at each of 1000 points...')

for i, time in enumerate(times):
    print(i + 1, '/1000 points calculated    \r', end='', sep='')
    dq = dq_calc(float(q_start[0]), float(q_start[1]), float(q_start[2]), time)
    q_start = q_start + dt * dq
    q_vals[:, i] = np.array([[float(q_start[0])],
                             [float(q_start[1])],
                             [float(q_start[2])]])[:, 0]
    position3[:, i] = np.array(p3_calc(float(q_start[0]),
                                     float(q_start[1]),
                                     float(q_start[2])))[:, 0]
    position2[:, i] = np.array(p2_calc(float(q_start[0]),
                                       float(q_start[1])))[:, 0]
    position1[:, i] = np.array(p1_calc(float(q_start[0])))[:, 0]

def animate(time):
    global position3

    # Update the line
    line.set_data(position3[0, :time], position3[1, :time])

    t_start = ax.transData

    # handle first link transform, initial position is locked, rotate about that position
    coords = t_start.transform([0, 0])
    t_change = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], q_vals[0, time] - np.pi / 2)
    t_end = t_start + t_change
    link1.set_xy([[-0.1, 0], [0.1, 0], [0.1, 0.9], [0.2, 0.9], [0.2, 1.1], [-0.2, 1.1], [-0.2, 0.9], [-0.1, 0.9]])
    link1.set_transform(t_end)

    # handle the movement of the second joint
    joint2.center = [position1[0, time], position1[1, time]]

    # handle second link transform
    link2.set_xy([[position1[0, time] - 0.075, position1[1, time] + q_vals[1, time]],
                  [position1[0, time] + 0.075, position1[1, time] + q_vals[1, time]],
                  [position1[0, time] + 0.075, position1[1, time] + q_vals[1, time] - 1.8],
                  [position1[0, time] - 0.075, position1[1, time] + q_vals[1, time] - 1.8]])
    coords = t_start.transform([position1[0, time], position1[1, time]])
    t_change = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], q_vals[0, time] - np.pi)
    t_end = t_start + t_change
    link2.set_transform(t_end)

    # handle the third joint transform
    joint3.center = [position2[0, time], position2[1, time]]

    # handle the third link transform
    link3.set_xy([[position2[0, time] - 0.075, position2[1, time]],
                  [position2[0, time] - 0.075, position2[1, time] + 0.8],
                  [position2[0, time], position2[1, time] + 1],
                  [position2[0, time] + 0.075, position2[1, time] + 0.8],
                  [position2[0, time] + 0.075, position2[1, time]]])
    coords = t_start.transform([position2[0, time], position2[1, time]])
    t_change = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], q_vals[0, time] + q_vals[2, time] - np.pi / 2)
    t_end = t_start + t_change
    link3.set_transform(t_end)

    # return the transformed links
    return link2, link1, circ, link3, joint3, joint2, line 

def init():
    line.set_data([], [])
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(position3.shape[1]), interval=12, init_func=init, blit=True, repeat=False)
plt.title('Trajectory of end point')
plt.show()
