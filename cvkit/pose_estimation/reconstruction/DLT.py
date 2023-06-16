# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:06:39 2018

@author: Jason
"""

# file DLTx.py version .1
'''
I found this code at https://www.mail-archive.com/floatcanvas@mithis.com/msg00513.html


Camera calibration and point reconstruction based on direct linear transformation (DLT).

The fundamental problem here is to find a mathematical relationship between the 
 coordinates  of a 3D point and its projection onto the image plane. The DLT 
 (a linear apporximation to this problem) is derived from modelling the object 
 and its projection on the image plane as a pinhole camera situation.
In simplistic terms, using the pinhole camera model, it can be found by similar 
 triangles the following relation between the image coordinates (u,v) and the 3D 
 point (X,Y,Z):
   [ u ]   [ L1  L2  L3  L4 ] [ X ]
   [ v ] = [ L5  L6  L7  L8 ] [ Y ]
   [ 1 ]   [ L9 L10 L11 L12 ] [ Z ]
                              [ 1 ]
The matrix L is kwnown as the camera matrix or camera projection matrix. For a 
 2D point (X,Y), the last column of the matrix doesn't exist. In fact, the L12 
 term (or L9 for 2D DLT) is not independent from the other parameters and then 
 there are only 11 (or 8 for 2D DLT) independent parameters in the DLT to be 
 determined.

DLT is typically used in two steps: 1. camera calibration and 2. object (point) 
 reconstruction.
The camera calibration step consists in digitizing points with known coordiantes 
 in the real space.
At least 4 points are necessary for the calibration of a plane (2D DLT) and at 
 least 6 points for the calibration of a volume (3D DLT). For the 2D DLT, at least
 one view of the object (points) must be entered. For the 3D DLT, at least 2 
 different views of the object (points) must be entered.
These coordinates (from the object and image(s)) are inputed to the DLTcalib 
 algorithm which  estimates the camera parameters (8 for 2D DLT and 11 for 3D DLT).
With these camera parameters and with the camera(s) at the same position of the 
 calibration step,  we now can reconstruct the real position of any point inside 
 the calibrated space (area for 2D DLT and volume for the 3D DLT) from the point 
 position(s) viewed by the same fixed camera(s). 

This code can perform 2D or 3D DLT with any number of views (views).
For 3D DLT, at least two views (views) are necessary.

There are more accurate (but more complex) algorithms for camera calibration that
 also consider lens distortion. For example, OpenCV and Tsai softwares have been
 ported to Python. However, DLT is classic, simple, and effective (fast) for 
 most applications.

About DLT, see: http://kwon3d.com/theory/dlt/dlt.html

This code is based on different implementations and teaching material on DLT 
 found in the internet.
'''

# Marcos Duarte - [EMAIL PROTECTED] - 04dec08

import numpy as np


def DLTrecon(nd, nc, Ls, uvs):
    '''
    Reconstruction of object point from image point(s) based on the DLT parameters.

    This code performs 2D or 3D DLT point reconstruction with any number of views (views).
    For 3D DLT, at least two views (views) are necessary.
    Inputs:
     nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
     nc is the number of views (views) used.
     Ls (array param_type) are the camera calibration parameters of each camera
      (is the output of DLTcalib function). The Ls parameters are given as columns
      and the Ls for different views as rows.
     uvs are the coordinates of the point in the image 2D space of each camera.
      The coordinates of the point are given as columns and the different views as rows.
    Outputs:
     xyz: point coordinates in space
    '''

    # Convert Ls to array:
    Ls = np.asarray(Ls)
    # Check the parameters:
    if Ls.ndim == 1 and nc != 1:
        raise ValueError(
            'Number of views (%d) and number of sets of camera calibration parameters (1) are different.' % (nc))
    if Ls.ndim > 1 and nc != Ls.shape[0]:
        raise ValueError(
            'Number of views (%d) and number of sets of camera calibration parameters (%d) are different.' % (
                nc, Ls.shape[0]))
    if nd == 3 and Ls.ndim == 1:
        raise ValueError('At least two sets of camera calibration parameters are needed for 3D point reconstruction.')

    if nc == 1:  # 2D and 1 camera (view), the simplest (and fastest) case
        # One could calculate inv(H) and input that to the code to speed up things if needed.
        # (If there is only 1 camera, this transformation is all Floatcanvas2 might need)
        Hinv = np.linalg.inv(Ls.reshape(3, 3))
        # Point coordinates in space:
        xyz = np.dot(Hinv, [uvs[0], uvs[1], 1])
        xyz = xyz[0:2] / xyz[2]
    else:
        M = []
        for i in range(nc):
            L = Ls[i, :]
            u, v = uvs[i][0], uvs[i][1]  # this indexing works for both list and numpy array
            if nd == 2:
                M.append([L[0] - u * L[6], L[1] - u * L[7], L[2] - u * L[8]])
                M.append([L[3] - v * L[6], L[4] - v * L[7], L[5] - v * L[8]])
            elif nd == 3:
                M.append([L[0] - u * L[8], L[1] - u * L[9], L[2] - u * L[10], L[3] - u * L[11]])
                M.append([L[4] - v * L[8], L[5] - v * L[9], L[6] - v * L[10], L[7] - v * L[11]])

        # Find the xyz coordinates:
        U, S, Vh = np.linalg.svd(np.asarray(M))
        # Point coordinates in space:
        xyz = Vh[-1, 0:-1] / Vh[-1, -1]
        xyz = [round(xyz[i], 4) for i in range(len(xyz))]
    return xyz


# Original Author: Yiwen Gu (yiweng@bu.edu)
def DLTdecon(Ls, xyz, nd=3, nc=2):
    '''
    Deconstruction of object point to image point(s) based on the DLT parameters.

    This code performs 3D or 2D DLT point reconstruction with any number of views (views).
    Inputs:
     nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
     nc is the number of views (views) used.
     Ls (array param_type) are the camera calibration parameters of each camera
      (is the output of DLTcalib function). The Ls parameters are given as columns
      and the Ls for different views as rows.
     xyz is point coordinates in object space, accept multiple points
    Outputs:
     uvs: the coordinates of the point in the image 2D space of each camera. 
         each inputs correspond to one 3D point
         every two columns is a pair of (u,v) in one camera view
    '''
    # Convert Ls to array:
    Ls = np.asarray(Ls)
    # Check the parameters:
    if Ls.shape[0] != nc:
        raise ValueError('Number of views does not match the number of sets of camera calibration parameters.')

    Ls = Ls.T
    lu = Ls[0:4]
    lv = Ls[4:8]
    ld = Ls[8:12]

    xyz1 = np.ones((len(xyz), 4))
    xyz1[:, 0:3] = xyz
    us = np.dot(xyz1, lu) / np.dot(xyz1, ld)
    vs = np.dot(xyz1, lv) / np.dot(xyz1, ld)

    uvs = np.empty((len(xyz), nc * 2))
    for i in range(nc):
        uvs[:, i * 2] = us[:, i]
        uvs[:, i * 2 + 1] = vs[:, i]
    return uvs
