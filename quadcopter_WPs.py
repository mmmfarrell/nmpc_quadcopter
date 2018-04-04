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

class Quadcopter:

    def __init__(self, waypoints, x_init=None):

        # Waypoints
        self.waypoints = waypoints
        self.current_wp = waypoints[0,:]
        self.wps_complete = 0
        self.reward = 0.
        self.wp_accept = 0.25 # acceptable distance to pass off waypoint

        # Dynamics params
        self.mass = 2.856
        self.linear_mu = 0.2
        self.angular_mu = 0.3
        self.ground_effect = [-55.3516, 181.8265, -203.9874, 85.3735, -7.6619]

        self.max_l = 6.5080
        self.max_m = 5.087
        self.max_n = 0.25
        self.max_F = 59.844

        self.Jx = 0.07
        self.Jy = 0.08
        self.Jz = 0.12

        self.J = np.diag([self.Jx, self.Jy, self.Jz])

        # Filter Outputs
        self.tau_up_l = 0.1904
        self.tau_up_m = 0.1904
        self.tau_up_n = 0.1644
        self.tau_up_F = 0.1644
        self.tau_down_l = 0.1904
        self.tau_down_m = 0.1904
        self.tau_down_n = 0.2164
        self.tau_down_F = 0.2164

        # Wind
        self.wind_n = 0.0
        self.wind_e = 0.0
        self.wind_d = 0.0

        # Definite intial conditions
        if x_init == None:
            self.x = np.zeros((12,1))

        # Init controllers
        self.roll_controller_ = PID(p=0.1, i=0.0, d=0.0)
        self.pitch_controller_ = PID(p=0.1, i=0.0, d=0.0)
        self.yaw_controller_ = PID(p=0.1, i=0.0, d=0.0)

        # Forces
        self.desired_forces_ = Forces()
        self.actual_forces_ = Forces()
        self.applied_forces_ = Forces()

    def force_and_moments(self, phi_c, theta_c, psi_rate_c, throttle, dt):

        # unpack state
        phi = self.x[6]
        theta = self.x[7]
        # psi = self.x[8]
        p = self.x[9]
        q = self.x[10]
        r = self.x[11]

        # Compute desired forces
        self.desired_forces_.l = self.roll_controller_.computePID(phi_c, phi, dt, p)
        self.desired_forces_.m = self.roll_controller_.computePID(theta_c, theta, dt, q)
        self.desired_forces_.n = self.roll_controller_.computePID(psi_rate_c, r, dt)
        self.desired_forces_.Fz = throttle*self.max_F

        # Calc acutal output with low-pass filters
        taul = self.tau_up_l if (self.desired_forces_.l > self.applied_forces_.l) else self.tau_down_l
        taum = self.tau_up_m if (self.desired_forces_.m > self.applied_forces_.m) else self.tau_down_m
        taun = self.tau_up_n if (self.desired_forces_.n > self.applied_forces_.n) else self.tau_down_n
        tauF = self.tau_up_F if (self.desired_forces_.Fz > self.applied_forces_.Fz) else self.tau_down_F

        # Calc alpha for filter
        alphal = dt/(taul + dt)
        alpham = dt/(taum + dt)
        alphan = dt/(taun + dt)
        alphaF = dt/(tauF + dt)

        # Apply discrete first-order filter
        self.applied_forces_.l = self.sat((1 - alphal)*self.applied_forces_.l + alphal*self.desired_forces_.l, self.max_l, -1.0*self.max_l)
        self.applied_forces_.m = self.sat((1 - alpham)*self.applied_forces_.m + alpham*self.desired_forces_.m, self.max_m, -1.0*self.max_m)
        self.applied_forces_.n = self.sat((1 - alphan)*self.applied_forces_.n + alphan*self.desired_forces_.n, self.max_n, -1.0*self.max_n)
        self.applied_forces_.Fz = self.sat((1 - alphaF)*self.applied_forces_.Fz + alphaF*self.desired_forces_.Fz, self.max_F, 0.0)

        # print "applied", self.applied_forces_.n
        # TODO add ground effect
        ground_effect = 0.0

        # TODO add Wind effect
        ur = 0.0
        vr = 0.0
        wr = 0.0

        # Apply other forces (i.e. wind)
        self.actual_forces_.Fx = -1.0*self.linear_mu*ur
        self.actual_forces_.Fy = -1.0*self.linear_mu*vr
        self.actual_forces_.Fz = -1.0*self.linear_mu*wr - self.applied_forces_.Fz - ground_effect
        self.actual_forces_.l = -1.0*self.angular_mu*p + self.applied_forces_.l
        self.actual_forces_.m = -1.0*self.angular_mu*q + self.applied_forces_.m
        self.actual_forces_.n = -1.0*self.angular_mu*r + self.applied_forces_.n
        # print "actual", self.actual_forces_.n
        k1 = self.derivatives(self.x)
        k2 = self.derivatives(self.x + (dt/2.)*k1)
        k3 = self.derivatives(self.x + (dt/2.)*k2)
        k4 = self.derivatives(self.x + (dt)*k3)

        self.x += (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)

        # Compute Reward

        # Compute distance to current waypoint
        dist = sqrt( (self.current_wp[0] - self.x[0])**2 + (self.current_wp[1] - self.x[1])**2 + (self.current_wp[2] - self.x[2])**2 )

        if (dist < self.wp_accept):
            if self.wps_complete == (len(self.waypoints[:,0])-1):
                self.reward = 1e10
            else:
                self.wps_complete += 1
                self.current_wp = self.waypoints[self.wps_complete, :]
                self.reward = self.wps_complete*1000.
        else:
            self.reward = self.wps_complete*1000. + (100.-dist) * 10.

    def derivatives(self, state):

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

        # pos = np.array([[pn], [pe], [pd]])
        pos = np.array([pn, pe, pd])
        vel = np.array([u, v, w])
        # att = np.array([phi, theta, psi])
        ang_vel = np.array([p, q, r])
        force = np.array([self.actual_forces_.Fx, self.actual_forces_.Fy, self.actual_forces_.Fz])
        torque = np.array([self.actual_forces_.l/self.Jx, self.actual_forces_.m/self.Jy, self.actual_forces_.n/self.Jz])
        force = np.reshape(force, (3,1))

        # position dot
        # Calc trigs
        cp = cos(phi)
        sp = sin(phi)
        ct = cos(theta)
        st = sin(theta)
        tt = tan(theta)
        cpsi = cos(psi)
        spsi = sin(psi)

        # calc rotation matrix
        # R_body_veh = np.array([[ct*cpsi, ct*spsi, -st],
        #                 [sp*st*cpsi-cp*spsi, sp*st*spsi+cp*cpsi, sp*ct],
        #                 [cp*st*cpsi+sp*spsi, cp*st*spsi-sp*cpsi, cp*ct]])

        grav = 9.81
        fg_body = np.array([-self.mass*grav*st, self.mass*grav*ct*sp, self.mass*grav*ct*cp])
        fg_body = np.reshape(fg_body, (3,1))

        rot_posdot = np.array([[ct*cpsi, sp*st*cpsi-cp*spsi, cp*st*cpsi+sp*spsi],
                        [ct*spsi, sp*st*spsi+cp*cpsi, cp*st*spsi-sp*cpsi],
                        [st, -sp*ct, -cp*ct]])


        pos_dot = np.matmul(rot_posdot, vel)
        vel_dot = np.array([r*v-q*w, p*w-r*u, q*u-p*v]) + (1./self.mass)*force + (1./self.mass)*fg_body
        vel_dot = np.reshape(vel_dot, (3,1))
        rot_attdot = np.array([[1., sp*tt, cp*tt], [0., cp, -sp], [0., (sp/ct), (cp/ct)]])
        att_dot = np.matmul(rot_attdot, ang_vel)
        ang_veldot = np.array([((self.Jy-self.Jz)/self.Jx)*q*r, ((self.Jz-self.Jx)/self.Jy)*p*r, ((self.Jx-self.Jy)/self.Jz)*p*q]) + torque

        # xdot
        xdot = np.zeros((12,1))
        xdot[0:3] = pos_dot
        xdot[2] = -xdot[2] # convert from hdot to pddot
        xdot[3:6] = vel_dot
        xdot[6:9] = att_dot
        xdot[9:] = ang_veldot

        return xdot

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


##############################
#### Main Function to Run ####
##############################
if __name__ == '__main__':

    # init path_manager_base object
    waypoints = np.array([[0., 0., -10.], [0., 0., 0.], [0., 0., -20.]])

    quad = Quadcopter(waypoints)
    plotter = QuadPlot()

    # Lets Fly :)
    dt = 0.01
    t = 0.0

    # Altitude Hold
    throttle_eq = quad.mass*9.81/quad.max_F
    alt_controller_ = PID(p=0.6, i=0.00, d=-0.4)
    alt_c = 10.0

    for i in range(10000):
        t += dt
        phi_c = 0.0
        theta_c = 0.0
        psirate_c = 0.0
        throttle_c = alt_controller_.computePID(alt_c, -quad.x[2], dt) + throttle_eq # eq force = -28.01736
        quad.force_and_moments(phi_c, theta_c, psirate_c, throttle_c, dt)
        if (i%10 == 0):
            plotter.plot(quad.x)
            print "--------------------"
            print "iteration #", i
            print "pos:", quad.x[0], quad.x[1], quad.x[2], quad.x[5]
            print "rot:", quad.x[6], quad.x[7], quad.x[8]
            print "time: ", t
            print "reward: ", quad.reward
        else:
            time.sleep(0.01)
        if quad.reward > 1000:
            alt_c = 0.
        if quad.reward > 2000:
            alt_c = 20.
