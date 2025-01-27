#!/usr/bin/env python3.10
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
from matplotlib.animation import FuncAnimation
import math
from functools import partial

#sys.stdin = open(0, 'r', buffering=0)
def read_line_from_stdin():
    line = sys.stdin.readline().strip()
    return line

def parse_stdin(line, x_vec, y_vec, z_vec, qw_vec, qx_vec, qy_vec, qz_vec):
    pattern = re.compile(r'Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+), qw = (-?\d+), qx = (-?\d+), qy = (-?\d+), qz = (-?\d+)')
    match = pattern.match(line)
    if match:
        x_vec.append(int(match.group(1))/100.0)
        y_vec.append(int(match.group(2))/100.0)
        z_vec.append(int(match.group(3))/100.0)
        qw_vec.append(int(match.group(4))/float((1 << 14)))
        qx_vec.append(int(match.group(5))/float((1 << 14)))
        qy_vec.append(int(match.group(6))/float((1 << 14)))
        qz_vec.append(int(match.group(7))/float((1 << 14)))
        return True
    return False

def parse_calib(line):
    pattern = re.compile(r'Calibration status: sys (-?\d+), gyro (-?\d+), accel (-?\d+), mag (-?\d+)')
    match = pattern.match(line)
    if match:
        sys = int(match.group(1))
        gyro = int(match.group(2))
        accel = int(match.group(3))
        mag = int(match.group(4))
        return {"sys":sys, "gyro":gyro, "accel":accel, "mag":mag}
    return False

def rotation_matrix_from_quaternion(w,x,y,z):
#this is post-multiplying matrix
#matlab quat2rotm is pre-multiplying
    rm = np.zeros((3,3))
    sqw,sqx,sqy,sqz = w*w,x*x,y*y,z*z
    wx = w*x
    wy = w*y
    wz = w*z
    xy = x*y
    xz = x*z
    yz = y*z

    rm[0][0] = sqw+sqx-sqy-sqz
    rm[0][1] = 2*(xy-wz)
    rm[0][2] = 2*(xz+wy)
    rm[1][0] = 2*(xy+wz)
    rm[1][1] = sqw-sqx+sqy-sqz
    rm[1][2] = 2*(yz-wx)
    rm[2][0] = 2*(xz-wy)
    rm[2][1] = 2*(yz+wx)
    rm[2][2] = sqw-sqx-sqy+sqz
    return rm
    
def quaternion_to_euler(q_vec):
#the result is consistent with matlab quat2angle ('zyx')
#Quaternion → Euler angles (z-y'-x'' intrinsic)
    qw,qx,qy,qz = q_vec
    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz
    
    yaw = np.arctan2(2*(qx*qy + qz*qw),(sqx - sqy - sqz + sqw))
    pitch = np.arcsin(-2*(qx*qz - qy*qw)/(sqx + sqy + sqz + sqw))
    roll = np.arctan2(2*(qy*qz + qx*qw),(-sqx - sqy + sqz + sqw))
    
    return yaw,pitch,roll

# Create figure for plotting
fig = plt.figure()
fig.suptitle('test title')
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

#fig2 = plt.figure(figsize=(8,5))
#ax3 = fig2.add_subplot(1, 1, 1, projection='3d')
#fig2.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)


t = []
x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec,yaw_vec,pitch_vec,roll_vec = [],[],[],[],[],[],[],[],[],[]

def animate(i, x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec, yaw_vec, pitch_vec, roll_vec):
    #TODO: use blit; return the actual lines (with ,) ; use set_ydata; set labels before hand
    window_size = 100

    #read the calibration status line
    line1 = read_line_from_stdin()
   
    status = parse_calib(line1)
    if status == False:
        return
    #read the measurements line
    line2 = read_line_from_stdin()
    if not parse_stdin(line2, x_vec, y_vec, z_vec, qw_vec, qx_vec, qy_vec, qz_vec):
        parse_stdin(line1, x_vec, y_vec, z_vec, qw_vec, qx_vec, qy_vec, qz_vec)

    #calcualte the angles
    qw,qx,qy,qz = qw_vec[-1],qx_vec[-1],qy_vec[-1],qz_vec[-1]
    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz
    yaw = np.arctan2(2*(qx*qy + qz*qw),(sqx - sqy - sqz + sqw))*180/math.pi
    pitch = np.arcsin(-2*(qx*qz - qy*qw)/(sqx + sqy + sqz + sqw))*180/math.pi
    roll = np.arctan2(2*(qy*qz + qx*qw),(-sqx - sqy + sqz + sqw))*180/math.pi

    #update the display vec
    yaw_vec.append(yaw)
    pitch_vec.append(pitch)
    roll_vec.append(roll)

    # Limit x and y lists
    #yaw_vec = yaw_vec[-window_size:]
    #pitch_vec = pitch_vec[-window_size:]
    #roll_vec = roll_vec[-window_size:]

    # sin
    ax1.clear()
    ax1.plot(yaw_vec, label="yaw")
    ax1.plot(pitch_vec, label="pitch")
    ax1.plot(roll_vec, label="roll")
    ax1.set_xlim(0, window_size)
    ax1.set_ylim(-180, 180)
    ax1.set_yticks([-180,-90,0,90,180])
    ax1.legend()
    
    ax2.clear()
    ax2.plot(x_vec[-window_size:], label="x_acc")
    ax2.plot(y_vec[-window_size:], label="y_acc")
    ax2.plot(z_vec[-window_size:], label="z_acc")
    ax2.set_xlim(0, window_size)
    ax2.set_ylim(-1,1)
    ax2.legend()
    
    ax3.clear()
    ax3.bar(status.keys(), status.values())
    ax3.set_ylim(0,3)
    ax3.set_yticks([0,1,2,3])
    
    if len(yaw_vec) == window_size:
        yaw_vec[:],pitch_vec[:],roll_vec[:] = [],[],[]
        x_vec[:], y_vec[:], z_vec[:] = [],[],[]
    

    #ax3.plot3D(xs, ys1, ys2, 'gray')
    #ax3.set_ylim(-1,1)
    #ax3.set_zlim(-1,1)

    #ax1.set_ylabel('sin')


ani = FuncAnimation(fig, animate, fargs=(x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec, yaw_vec, pitch_vec,roll_vec), interval=100, cache_frame_data = False, blit=False)
plt.show()