from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from math import *
import time

class QuadPlot(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)
        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        self.w.show()
        self.w.setWindowTitle('Quadcopter Simulation')

        ## create three grids, add each to the view
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        self.w.addItem(xgrid)
        self.w.addItem(ygrid)
        self.w.addItem(zgrid)

        ## rotate x and y grids to face the correct direction
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)

        xgrid.scale(1.0, 1.0, 1.0)
        ygrid.scale(1.0, 1.0, 1.0)
        zgrid.scale(1.0, 1.0, 1.0)

        rad = 0.5

        arm_length = 1.0 + rad
        arm_width = 0.25/2.
        arm_height = 0.25/2.

        arm = np.array([
            [-arm_width, -0, -arm_height],
            [-arm_width, -0, arm_height],
            [-arm_width, arm_length, -arm_height],
            [-arm_width, arm_length, arm_height],
            [arm_width, arm_length, -arm_height],
            [arm_width, arm_length, arm_height],
            [arm_width, -0, -arm_height],
            [arm_width, -0, arm_height],
        ])

        rot_z1 = np.array([[cos(np.pi/4), -sin(np.pi/4), 0], [sin(np.pi/4), cos(np.pi/4), 0], [0., 0., 1.]])
        rot_z2 = np.array([[cos(3*np.pi/4), -sin(3*np.pi/4), 0], [sin(3*np.pi/4), cos(3*np.pi/4), 0], [0., 0., 1.]])
        rot_z3 = np.array([[cos(-np.pi/4), -sin(-np.pi/4), 0], [sin(-np.pi/4), cos(-np.pi/4), 0], [0., 0., 1.]])
        rot_z4 = np.array([[cos(-3*np.pi/4), -sin(-3*np.pi/4), 0], [sin(-3*np.pi/4), cos(-3*np.pi/4), 0], [0., 0., 1.]])

        verts1 = np.matmul(arm, rot_z1)
        verts2 = np.matmul(arm, rot_z2)
        verts3 = np.matmul(arm, rot_z3)
        verts4 = np.matmul(arm, rot_z4)

        faces = np.array([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
            [7, 0, 1],
            [1, 3, 5],
            [1, 7, 5],
            [0, 2, 4],
            [0, 6, 4]
        ])
        colors_back = np.array([
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3],
            [1, 0, 0, 0.3]
        ])
        colors_front = np.array([
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3],
            [1, 1, 1, 0.3]
        ])
        #
        ## Mesh item will automatically compute face normals.
        self.arm1 = gl.GLMeshItem(vertexes=verts1, faces=faces, faceColors=colors_back, smooth=False)
        self.arm2 = gl.GLMeshItem(vertexes=verts2, faces=faces, faceColors=colors_back, smooth=False)
        self.arm3 = gl.GLMeshItem(vertexes=verts3, faces=faces, faceColors=colors_front, smooth=False)
        self.arm4 = gl.GLMeshItem(vertexes=verts4, faces=faces, faceColors=colors_front, smooth=False)
        self.w.addItem(self.arm1)
        self.w.addItem(self.arm2)
        self.w.addItem(self.arm3)
        self.w.addItem(self.arm4)

        md = gl.MeshData.sphere(rows=10, cols=10, radius=rad)
        self.body = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=True, drawEdges=True, edgeColor=(1,0,0,1), color=(1,0,0,1) )
        self.w.addItem(self.body)

        # Keep track of my own state
        self.x = np.zeros((12,))

    def plot(self, x_new):
        dn = - self.x[0] + x_new[0]
        de = - self.x[1] + x_new[1]
        dd = - self.x[2] + x_new[2]
        dr = - self.x[6] + x_new[6]
        dp = - self.x[7] + x_new[7]
        dy = - self.x[8] + x_new[8]
        self.update(dn, de, dd, dr*180./np.pi, dp*180./np.pi, dy*180./np.pi)
        self.x = x_new.copy()

    def update(self, dn, de, dd, dr, dp, dy):

        # GL Frame is SEU
        # x = -pn, y = pe, z = -pd
        # Translate
        self.body.translate(-dn, de, -dd)
        self.arm1.translate(-dn, de, -dd)
        self.arm2.translate(-dn, de, -dd)
        self.arm3.translate(-dn, de, -dd)
        self.arm4.translate(-dn, de, -dd)

        # Rotate roll
        self.arm1.rotate(-dr, 1, 0, 0, local=True)
        self.arm2.rotate(-dr, 1, 0, 0, local=True)
        self.arm3.rotate(-dr, 1, 0, 0, local=True)
        self.arm4.rotate(-dr, 1, 0, 0, local=True)

        # Rotate yaw
        self.arm1.rotate(dp, 0, 1, 0, local=True)
        self.arm2.rotate(dp, 0, 1, 0, local=True)
        self.arm3.rotate(dp, 0, 1, 0, local=True)
        self.arm4.rotate(dp, 0, 1, 0, local=True)

        # Rotate roll
        self.arm1.rotate(-dy, 0, 0, 1, local=True)
        self.arm2.rotate(-dy, 0, 0, 1, local=True)
        self.arm3.rotate(-dy, 0, 0, 1, local=True)
        self.arm4.rotate(-dy, 0, 0, 1, local=True)

        # Update Plot
        self.w.show()
        self.app.processEvents()
        time.sleep(0.01)


if __name__ == '__main__':
    # import sys
    plotter = QuadPlot()
    for i in range(1000):
        plotter.update(0.00, 0, 0, 0, 0.0, 0.1)
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()
