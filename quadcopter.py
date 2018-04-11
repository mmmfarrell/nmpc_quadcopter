'''
    This file defines a quadcopter class that simulates quadcopter dynamics.

    # State vector x = (12,1)
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r

    # Actions u = (4,1)

'''
import numpy as np
from math import *
from simple_pid import PID
import time
from Vizualize import QuadPlot
from nmpc_control import NMPC

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Quadcopter:

    def __init__(self, x_init=None):

        # Dynamics params
        self.mass = 0.8
        # self.linear_mu = 0.2
        # self.angular_mu = 0.3
        # self.ground_effect = [-55.3516, 181.8265, -203.9874, 85.3735, -7.6619]

        # self.max_l = 6.5080
        # self.max_m = 5.087
        # self.max_n = 0.25
        # self.max_F = 59.844

        self.Jx = 0.0224
        self.Jy = 0.0224
        self.Jz = 0.0436

        self.J = np.diag([self.Jx, self.Jy, self.Jz])

        self.L = 0.165 # m
        self.c = 0.002167 # m
        self.g = 9.8
        self.lambdaF = 50

        # Make constant Bc matrix
        self.Bc = np.zeros((12, 4))
        self.Bc[5, 0] = -1.0/self.mass
        self.Bc[5, 1] = -1.0/self.mass
        self.Bc[5, 2] = -1.0/self.mass
        self.Bc[5, 3] = -1.0/self.mass
        self.Bc[9, 1] = -self.L/self.Jx
        self.Bc[9, 3] = self.L/self.Jx
        self.Bc[10, 0] = self.L/self.Jy
        self.Bc[10, 2] = -self.L/self.Jy
        self.Bc[11, 0] = -self.c/self.Jz
        self.Bc[11, 1] = self.c/self.Jz
        self.Bc[11, 2] = -self.c/self.Jz
        self.Bc[11, 3] = self.c/self.Jz

        # # Filter Outputs
        # self.tau_up_l = 0.1904
        # self.tau_up_m = 0.1904
        # self.tau_up_n = 0.1644
        # self.tau_up_F = 0.1644
        # self.tau_down_l = 0.1904
        # self.tau_down_m = 0.1904
        # self.tau_down_n = 0.2164
        # self.tau_down_F = 0.2164

        # # Wind
        # self.wind_n = 0.0
        # self.wind_e = 0.0
        # self.wind_d = 0.0

        # Definite intial conditions
        if x_init == None:
            self.x = np.zeros((12,1))

        # # Init controllers
        # self.roll_controller_ = PID(p=0.1, i=0.0, d=0.0)
        # self.pitch_controller_ = PID(p=0.1, i=0.0, d=0.0)
        # self.yaw_controller_ = PID(p=0.1, i=0.0, d=0.0)

        # # Forces
        # self.desired_forces_ = Forces()
        # self.actual_forces_ = Forces()
        # self.applied_forces_ = Forces()

    def force_and_moments(self, u, dt):

        # k1 = self.derivatives(self.x, u)
        # k2 = self.derivatives(self.x + (dt/2.)*k1, u)
        # k3 = self.derivatives(self.x + (dt/2.)*k2, u)
        # k4 = self.derivatives(self.x + (dt)*k3, u)

        # self.x += (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)

        for i in range(0, 10):
            self.x += (dt/10.)*self.derivatives(self.x, u)

        # A, B = self.get_SS()
        # self.x += dt*(A.dot(quad.x) + B.dot(u))

    def derivatives(self, state, F):

        # Unpack state
        pn = state[0]
        pe = state[1]
        pd = state[2]
        u = state[3]
        v = state[4]
        w = state[5]
        phi = state[6]
        theta = state[7]
        psi = state[8]
        p = state[9]
        q = state[10]
        r = state[11]

        pos = np.array([pn, pe, pd])
        vel = np.array([u, v, w])
        ang_vel = np.array([p, q, r])
        force = F

        # Calc trigs
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        st = sin(theta)
        tt = tan(theta)
        cpsi = cos(psi)
        spsi = sin(psi)

        # Position dot
        # calc rotation matrix
        R_bi = np.array([[ct*cpsi, ct*spsi, -st],
                        [sp*st*cpsi-cp*spsi, sp*st*spsi+cp*cpsi, sp*ct],
                        [cp*st*cpsi+sp*spsi, cp*st*spsi-sp*cpsi, cp*ct]])

        pos_dot = np.matmul(R_bi.T, vel)

        # Velocity dot
        ehat3 = np.zeros((3,1))
        ehat3[2] = 1.0

        vel_dot = -np.cross(ang_vel.T, vel.T).T + np.matmul(R_bi, (self.g * ehat3))

        # Angles dot
        W = np.array([[1.0, sp*st/ct, cp*st/ct],
                      [0.0, cp, -sp],
                      [0.0, sp/ct, cp/ct]])

        ang_dot = np.matmul(W, ang_vel)

        # angvel_dot
        angvel_dot = np.matmul(np.linalg.inv(self.J), np.cross(-ang_vel.T, np.matmul(self.J, ang_vel).T).T)

        # force xdot
        force_xdot = np.matmul(self.Bc, force)

        # xdot
        xdot = np.zeros((12,1))
        xdot[0:3] = pos_dot
        xdot[3:6] = vel_dot
        xdot[6:9] = ang_dot
        xdot[9:] = angvel_dot

        xdot += force_xdot

        return xdot
    
    def get_SS_linear(self):
        # Unpack state
        # pn = self.x[0]
        # pe = self.x[1]
        # pd = self.x[2]
        u = 0.0
        v = 0.0
        w = 0.0
        phi = 0.0
        theta = 0.0
        psi = 0.0
        p = 0.0
        q = 0.0
        r = 0.0

        # Calc trigs
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        st = sin(theta)
        tt = tan(theta)
        cpsi = cos(psi)
        spsi = sin(psi)

        # Compute Ac.
        A = np.zeros((12, 12))

        R_bi = np.array([[ct*cpsi, ct*spsi, -st],
                        [sp*st*cpsi-cp*spsi, sp*st*spsi+cp*cpsi, sp*ct],
                        [cp*st*cpsi+sp*spsi, cp*st*spsi-sp*cpsi, cp*ct]])
        A[0:3, 3:6] = R_bi.T

        A22 = np.array([[0.0, r/2.0, -q/2.0],
                        [-r/2.0, 0.0, p/2.0],
                        [q/2.0, -p/2.0, 0.0]])
        A[3:6, 3:6] = A22

        A23 = np.array([[0.0, -self.g*(1.0 - theta*theta/6.0 + (theta**4)/120.), 0.0],
                        [self.g*ct*(1.0 - phi*phi/6.0 + (phi**4)/120.), 0.0, 0.0],
                        [self.g*((ct + 1.0)/2.0)*(-phi/2.0 + (phi**3)/24.0 - (phi**5)/720.0), 
                            self.g*((cp + 1.0)/2.0)*(-theta/2.0 + (theta**3)/24.0 - (theta**5)/720.),
                            0.0]])
        A[3:6, 6:9] = A23

        A24 = np.array([[0.0, -w/2.0, v/2.0],
                        [w/2.0, 0.0, -u/2.0],
                        [-v/2.0, u/2.0, 0.0]])
        A[3:6, 9:12] = A24

        W = np.array([[1.0, sp*st/ct, cp*st/ct],
                      [0.0, cp, -sp],
                      [0.0, sp/ct, cp/ct]])
        A[6:9, 9:12] = W

        A44 = np.array([[0.0, (self.Jy - self.Jz)*r/(2.0*self.Jx), (self.Jy - self.Jz)*q/(2.0*self.Jx)],
                        [(self.Jz - self.Jx)*r/(2.0*self.Jy), 0.0, (self.Jz - self.Jx)*p/(2.0*self.Jy)],
                        [(self.Jx - self.Jy)*q/(2.0*self.Jz), (self.Jx - self.Jy)*p/(2.0*self.Jz), 0.0]])
        A[9:12, 9:12] = A44

        B = np.copy(self.Bc)

        return A, B

    def get_SS(self):
        # Unpack state
        pn = self.x[0]
        pe = self.x[1]
        pd = self.x[2]
        u = self.x[3]
        v = self.x[4]
        w = self.x[5]
        phi = self.x[6]
        theta = self.x[7]
        psi = self.x[8]
        p = self.x[9]
        q = self.x[10]
        r = self.x[11]

        # Calc trigs
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        st = sin(theta)
        tt = tan(theta)
        cpsi = cos(psi)
        spsi = sin(psi)

        # Compute Ac.
        A = np.zeros((12, 12))

        R_bi = np.array([[ct*cpsi, ct*spsi, -st],
                        [sp*st*cpsi-cp*spsi, sp*st*spsi+cp*cpsi, sp*ct],
                        [cp*st*cpsi+sp*spsi, cp*st*spsi-sp*cpsi, cp*ct]])
        A[0:3, 3:6] = R_bi.T

        A22 = np.array([[0.0, r/2.0, -q/2.0],
                        [-r/2.0, 0.0, p/2.0],
                        [q/2.0, -p/2.0, 0.0]])
        A[3:6, 3:6] = A22

        A23 = np.array([[0.0, -self.g*(1.0 - theta*theta/6.0 + (theta**4)/120.), 0.0],
                        [self.g*ct*(1.0 - phi*phi/6.0 + (phi**4)/120.), 0.0, 0.0],
                        [self.g*((ct + 1.0)/2.0)*(-phi/2.0 + (phi**3)/24.0 - (phi**5)/720.0), 
                            self.g*((cp + 1.0)/2.0)*(-theta/2.0 + (theta**3)/24.0 - (theta**5)/720.),
                            0.0]])
        A[3:6, 6:9] = A23

        A24 = np.array([[0.0, -w/2.0, v/2.0],
                        [w/2.0, 0.0, -u/2.0],
                        [-v/2.0, u/2.0, 0.0]])
        A[3:6, 9:12] = A24

        W = np.array([[1.0, sp*st/ct, cp*st/ct],
                      [0.0, cp, -sp],
                      [0.0, sp/ct, cp/ct]])
        A[6:9, 9:12] = W

        A44 = np.array([[0.0, (self.Jy - self.Jz)*r/(2.0*self.Jx), (self.Jy - self.Jz)*q/(2.0*self.Jx)],
                        [(self.Jz - self.Jx)*r/(2.0*self.Jy), 0.0, (self.Jz - self.Jx)*p/(2.0*self.Jy)],
                        [(self.Jx - self.Jy)*q/(2.0*self.Jz), (self.Jx - self.Jy)*p/(2.0*self.Jz), 0.0]])
        A[9:12, 9:12] = A44

        B = np.copy(self.Bc)

        return A, B

    def sat(self, x, _max, _min):
        if (x > _max):
            # print "SAT MAX"
            return _max
        elif (x < _min):
            # print "SAT MIN"
            return _min
        else:
            return x

class Forces:
    def __init__(self):
        self.l = 0.0
        self.m = 0.0
        self.n = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0

def make_ref_traj(dt):

    # choose trajetory
    traj = 4

    if traj == 0:
        # hold 0 position
        T = 10.0
        steps = int(T/dt)
        xr = np.zeros((12, 30*steps))
        # xr[2, :] = -10.0
        # xr[2, 2*steps:] = 0.0
    elif traj == 1:
        # Straight up trajectory for 10m, 10seconds
        T = 10.0
        steps = int(T/dt)

        north_steps = np.linspace(0.0, 10.0, steps)

        xr = np.zeros((12, 30*steps))
        xr[2, 0:steps] = -north_steps
        # xr[5, 0:steps] = -1.0
        xr[2, steps:] = -10.0
    elif traj == 2:
        # Straight north trajectory for 10m, 10seconds
        T = 6.0
        steps = int(T/dt)

        north_steps = np.linspace(0.0, 10.0, steps)

        xr = np.zeros((12, steps + 50))
        xr[0, 0:steps] = north_steps
        # xr[7, 0:steps] = np.pi/6.0
        xr[0, steps:] = 10.0
    elif traj == 3:
        # Straight north trajectory for 10m, 10seconds
        T = 6.0
        steps = int(T/dt)

        north_steps = np.linspace(0.0, 10.0, steps)

        xr = np.zeros((12, 5*steps + 50))
        xr[0, :] = 10.0
        xr[0, 0:steps] = north_steps

        # Make a ciricle of Radius 10
        theta = np.linspace(0.0, 2*np.pi, 4*steps)
        north = np.zeros_like(theta)
        east = np.zeros_like(theta)

        for i in range(len(theta)):
            north[i] = 10.0 * cos(theta[i])
            east[i] = 10.0 * sin(theta[i])

        xr[0, steps:5*steps] = north
        xr[1, steps:5*steps] = east
        # xr[8, steps:5*steps] = theta
    elif traj == 4:
        # Spiral
        T = 10.0
        steps = int(T/dt)

        north_steps = np.linspace(0.0, 10.0, steps)

        xr = np.zeros((12, 5*steps + 50))
        xr[0, :] = 10.0
        # xr[2, :] = -10.0
        xr[0, 0:steps] = north_steps

        # Make a ciricle of Radius 10
        theta = np.linspace(0.0, 2*np.pi, 4*steps)
        north = np.zeros_like(theta)
        east = np.zeros_like(theta)
        down = np.linspace(0.0, -10.0, 4*steps)

        for i in range(len(theta)):
            north[i] = 10.0 * cos(theta[i])
            east[i] = 10.0 * sin(theta[i])

        xr[0, steps:5*steps] = north
        xr[1, steps:5*steps] = east
        xr[2, steps:5*steps] = down
        xr[2, 5*steps:] = -10.0
        # xr[8, steps:5*steps] = theta

    return xr


##############################
#### Main Function to Run ####
##############################
if __name__ == '__main__':

    # init path_manager_base object
    quad = Quadcopter()
    plotter = QuadPlot()
    controller = NMPC()

    # Lets Fly :)
    dt = 0.05
    t = 0.0

    xr = make_ref_traj(dt)

    # Altitude Hold
    throttle_eq = quad.mass*9.8/4.0 

    u_eq = np.array([[throttle_eq],
                     [throttle_eq],
                     [throttle_eq],
                     [throttle_eq]])

    # Plot results
    PLOT = True
    ANIMATE = True
    if PLOT:
        x_result = np.zeros((12,1))

    for i in range(len(xr[0,:]) - 25):
        t += dt

        # Compute control
        start = time.time()
        u = controller.compute_control(quad, u_eq, xr[:,i:i+26])
        end = time.time()
        print "time to compute: ", end - start
        # print "control u: ", u

        # Take a step with control
        quad.force_and_moments(u + u_eq, dt)

        if PLOT:
            # x_result.append(quad.x)
            x_result = np.hstack((x_result, quad.x))
            # print "results shape: ", x_result.shape

        if True:#(i%10 == 0):
            plotter.plot(quad.x)
            print "--------------------"
            print "iteration #", i
            print "time: ", t
            print "reference loc: ", xr[:,i]
            print "pos:", quad.x[0], quad.x[1], quad.x[2], quad.x[5]
            print "rot:", quad.x[6], quad.x[7], quad.x[8]
            print "Control: ", u
            time.sleep(0.01)

    if PLOT:
        x_result_north = x_result[0,:]
        x_result_east = x_result[1,:]
        x_result_down = x_result[2,:]
        x_r_north = xr[0,:]
        x_r_east = xr[1,:]
        x_r_down = xr[2,:]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(x_r_east, x_r_north, -x_r_down, 'b')
        plt.plot(x_result_east, x_result_north, -x_result_down, 'r')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.axis('equal')
        plt.legend(['Trajectory', 'Result'])
        plt.show()

    if ANIMATE:
        for i in range(len(x_result[0,:])):
            quad.x = x_result[:,i]
            plotter.plot(quad.x)
            time.sleep(dt)


