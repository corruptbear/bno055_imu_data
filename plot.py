#!/usr/bin/env python3.10
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import os
from os import path
pwd = path.dirname(path.realpath(__file__))

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin/'
print(os.environ['PATH'])
plt.rcParams['text.usetex'] = True


def parse_linear_acc_log(filename):
    with open(filename,"r") as f:
        lines = f.readlines()
        x_vec,y_vec,z_vec = [],[],[]
        pattern = re.compile(r'Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+)')

        for line in lines:
            match = pattern.match(line)
            if match:
                x_vec.append(int(match.group(1)))
                y_vec.append(int(match.group(2)))
                z_vec.append(int(match.group(3)))
    return x_vec,y_vec,z_vec

def parse_linear_acc_quaternion_log(filename):
    with open(filename,"r") as f:
        lines = f.readlines()
        x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec = [],[],[],[],[],[],[]
        pattern = re.compile(r'Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+), qw = (-?\d+), qx = (-?\d+), qy = (-?\d+), qz = (-?\d+)')

        for line in lines:
            match = pattern.match(line)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                z = int(match.group(3))
                #ugly fix the sign bit flipping problem
                #it's safe because the maximum range for acc is 16g, which means ~16000 is the maximum value when using mg/s^2 as unit
                if x>16000:
                    x=x-32768
                if x<-16000:
                    x=x+32768
                if y>16000:
                    y=y-32768
                if y<-16000:
                    y=y+32768
                if z>16000:
                    z=z-32768
                if z<-16000:
                    z=z+32768
                x_vec.append(x)
                y_vec.append(y)
                z_vec.append(z)
                qw_vec.append(int(match.group(4))/float((1 << 14)))
                qx_vec.append(int(match.group(5))/float((1 << 14)))
                qy_vec.append(int(match.group(6))/float((1 << 14)))
                qz_vec.append(int(match.group(7))/float((1 << 14)))
    return np.array(x_vec),np.array(y_vec),np.array(z_vec),np.array(qw_vec),np.array(qx_vec),np.array(qy_vec),np.array(qz_vec)

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
    
def quaternion_to_euler(qw,qx,qy,qz):
    """
    the result is consistent with matlab quat2angle ('zyx') 
    Quaternion → Euler angles (z-y'-x'' intrinsic)
    """

    #ensure that the norm of the quarternions are 1
    q_norm =  np.sqrt(qw*qw+qx*qx+qy*qy+qz*qz)
    assert np.mean(abs(np.ones_like(q_norm) - q_norm))<0.01

    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz

    yaw = np.arctan2(2*(qx*qy + qz*qw),(sqx - sqy - sqz + sqw))
    pitch = np.arcsin(-2*(qx*qz - qy*qw)/(sqx + sqy + sqz + sqw))
    roll = np.arctan2(2*(qy*qz + qx*qw),(-sqx - sqy + sqz + sqw))

    return yaw,pitch,roll


def reconstruct_pos_double_int_linear_acc_quaternion(x_vec,y_vec,z_vec,qw,qx,qy,qz, sampling_rate = 50, scaling_factor = 100.0):
    """
    double integration
    """
    yaw, pitch, roll = quaternion_to_euler(qw,qx,qy,qz)
    x_vec,y_vec,z_vec = x_vec/scaling_factor,y_vec/scaling_factor,z_vec/scaling_factor
    SAMPLE_INTERVAL = 1.0/sampling_rate
    vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
    position_x, position_y, position_z = 0.0, 0.0, 0.0
    positions_x,positions_y,positions_z = [],[],[]
    projected_x_vec,projected_y_vec,projected_z_vec = [],[],[]
    vels_x, vels_y, vels_z = [],[],[]
    
    
    acc_norm_vec = np.sqrt(x_vec*x_vec+y_vec*y_vec+z_vec*z_vec)
    print("acc norm:", acc_norm_vec)
    
    for i in range(len(x_vec)):
        w = qw[i]
        x = qx[i]
        y = qy[i]
        z = qz[i]
        r = rotation_matrix_from_quaternion(w,x,y,z)

        #rotate the acc to world frame
        projected_acc = r.dot(np.array([x_vec[i],y_vec[i],z_vec[i]]))
        
        projected_x_vec.append(projected_acc[0])
        projected_y_vec.append(projected_acc[1])
        projected_z_vec.append(projected_acc[2])
        
        #projected_x_vec.append(x_vec[i])
        #projected_y_vec.append(y_vec[i])
        #projected_z_vec.append(z_vec[i])

        #avg
        avg_x = vel_x + 0.5 * SAMPLE_INTERVAL * projected_acc[0]
        avg_y = vel_y + 0.5 * SAMPLE_INTERVAL * projected_acc[1]
        avg_z = vel_z + 0.5 * SAMPLE_INTERVAL * projected_acc[2]
        
        vel_x = vel_x + SAMPLE_INTERVAL * projected_acc[0]
        vel_y = vel_y + SAMPLE_INTERVAL * projected_acc[1]
        vel_z = vel_z + SAMPLE_INTERVAL * projected_acc[2]
        
        vels_x.append(vel_x)
        vels_y.append(vel_y)
        vels_z.append(vel_z)
        
        position_x = position_x + SAMPLE_INTERVAL * avg_x
        position_y = position_y + SAMPLE_INTERVAL * avg_y
        position_z = position_z + SAMPLE_INTERVAL * avg_z

        positions_x.append(position_x)
        positions_y.append(position_y)
        positions_z.append(position_z)
        
    print(len(yaw),len(positions_x))
    print("mean x velocity:",np.mean(projected_x_vec),"mean y velocity:",np.mean(projected_y_vec),"mean z velocity:",np.mean(projected_z_vec))
    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)


    ax1.plot(yaw*180/math.pi,label="yaw")
    ax1.plot(pitch*180/math.pi,label="pitch")
    ax1.plot(roll*180/math.pi,label="roll")
    ax1.set_ylim(-180, 180)
    ax1.set_yticks([-180,-90,0,90,180])
    ax1.set_ylabel(r"degree")
    ax1.legend()
    
    ax2.plot(x_vec, label="x_acc")
    ax2.plot(y_vec, label="y_acc")
    ax2.plot(z_vec, label="z_acc")
    ax2.set_ylabel(r"$m/s^2$")
    #ax2.plot(acc_norm_vec, label="norm_acc")
    ax2.legend()
    
    ax3.plot(projected_x_vec, label="proj_x_acc")
    ax3.plot(projected_y_vec, label="proj_y_acc")
    ax3.plot(projected_z_vec, label="proj_z_acc")
    ax3.legend()
    
    ax4.plot(vels_x, label="proj_vel_x")
    ax4.plot(vels_y, label="proj_vel_y")
    ax4.plot(vels_z, label="proj_vel_z")
    ax4.set_ylabel(r"$m/s$")
    ax4.legend()
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot3D(positions_x, positions_y, positions_z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


#x_vec,y_vec,z_vec = parse_linear_acc_log("linear_acc_50Hz.txt")
#reconstruct(x_vec[:700], y_vec[:700], z_vec[:700], sampling_rate = 50, scaling_factor = 100 )
q = [0.71990966796875,0.0728759765625,0.088134765625,-0.6845703125]
print(quaternion_to_euler(*q))
print(rotation_matrix_from_quaternion(0.9659,0.2588,0,0))



#calibrated_linear_50HZ.txt: back and forth; largely linear
#calibrated_static_50HZ.txt: highlights drifting
x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec = parse_linear_acc_quaternion_log(path.join(pwd,"logs/calibrated_linear_50HZ.txt"))
#x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec = parse_linear_acc_quaternion_log(path.join(pwd,"calibrated_static_50HZ.txt"))
reconstruct_pos_double_int_linear_acc_quaternion(x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec, sampling_rate = 50)

#print(qw_vec[0],qx_vec[0],qy_vec[0],qz_vec[0])
#rotation_matrix_from_quaternion(qw_vec[0],qx_vec[0],qy_vec[0],qz_vec[0])