"""
Inverted Pendulum Example

copied from Atsushi Sakai

"""

import numpy as np
import math
from math import *
import time
import cvxpy

import control

class NMPC(object):
    """Nonlinear MPC for inverted pendulum. """

    def __init__(self):
        # State space info
        self.nx = 12 # number of states
        self.nu = 4 # number of control inputs
        
        # Cost function parameters
        self.Q = np.eye(12)
        # self.Q = np.zeros((12,12))
        self.Q[0, 0] = 1.0
        self.Q[1, 1] = 1.0
        self.Q[2, 2] = 1.0
        # self.Q = np.diag([0.0, 1.0, 1.0, 0.0]) # Matrix for quadratic cost function with state
        self.R = np.diag([0.001, 0.001, 0.001, 0.001]) # Matrix for quadratic cost function with inputs

        # Time parameters
        self.T = 25 # horizon length
        self.delta_t = 0.01 # time steps

        # Set reference position
        self.xr = np.zeros((12,1))
        self.xr[2] = 10.0

    def compute_control(self, quad, u_eq):
        try:
            # Compute control given current state
            ox, dx, otheta, dtheta, ou = self.mpc_control(quad, u_eq)
            # print "ou: ", ou
            u = ou[:, 0]
        except:
            u = np.zeros((4,1))
            print "ERROR mpc control failed, repeating previous input"

        return u

    def mpc_control(self, quad, u_eq):

        # Define optimzation variables
        x = cvxpy.Variable(self.nx, self.T + 1) # size of variable 4 x (T+1)
        u = cvxpy.Variable(self.nu, self.T) # size of variable 1 x T

        # Get SS representation of dynamics
        A, B = quad.get_SS()
        C = np.identity(12)
        D = np.zeros((12,4))
        cont_sys = control.StateSpace(A, B, C, D)
        discrete_sys = control.matlab.c2d(cont_sys, self.delta_t)

        A = discrete_sys.A
        B = discrete_sys.B

        cost = 0.0
        constr = []

        # Loop through horizon
        for t in range(self.T):
            cost += cvxpy.quad_form(x[:, t + 1] - self.xr, self.Q) # This does xT * Q * x
            cost += cvxpy.quad_form(u[:, t], self.R) # This does uT * R * u
            constr += [x[:, t + 1] == A * x[:, t] + B * (u[:, t] + u_eq)] # Contraint to follow dynamics
            # constr += [x[:, t + 1] == self.pend.get_discrete_step(x[:, t], u[:, t], self.delta_t)] # Contraint to follow dynamics

            # # Constrain x position to be between -5, 5
            # constr += [(x[0, t + 1]) <= 5.0]
            # constr += [(x[0, t + 1]) >= -5.0]

            # # Constrain u force 
            # constr += [u[:, t + 1] <= 10.0]
            # constr += [u[:, t + 1] >= -10.0]
        constr += [x[:, 0] == quad.x] # Contraint for initial conditions

        # Create cvxpy optimization problem
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)

        start = time.time()
        prob.solve(verbose=False) # Solve optimization
        elapsed_time = time.time() - start
        # print("calc time:{0} [sec]".format(elapsed_time))

        # If optimal solution reached
        if prob.status == cvxpy.OPTIMAL:
            ox = self.get_nparray_from_matrix(x.value[0, :]) # result for x for all timesteps
            dx = self.get_nparray_from_matrix(x.value[1, :])
            theta = self.get_nparray_from_matrix(x.value[2, :])
            dtheta = self.get_nparray_from_matrix(x.value[3, :])

            # print "u val: ", u.value
            # print "shape: ", u.value.shape
            # ou = self.get_nparray_from_matrix(u.value[:, :])
            ou = u.value

        else:
            print("ERROR: Optimization not OPTIMAL")
            return 0, 0, 0, 0, 0


        return ox, dx, theta, dtheta, ou

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def plot(self):
        self.pend.plot()

def main():
    # Instantiate controller
    mpc = NMPC()

    # Constant force and dt
    u = 0.
    dt = 0.05

    # Loop through timesteps
    for i in range(0, 500):
        # Compute control
        u = mpc.compute_control(u)

        # Simulate dynamics
        mpc.pend.simulate(u, dt)

        # plot pend
        mpc.plot()

if __name__ == '__main__':
    main()
