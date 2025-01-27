import multiprocessing
import sys
from matplotlib.animation import FuncAnimation
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import signal
from filelock import Timeout, FileLock
import time
import subprocess

#-----------------this is for plot BLE imu data from single device-------------------------

pwd = os.path.dirname(os.path.realpath(__file__))

def handle_interrupt(signal, frame):
    print("\nReceived Ctrl + C, exiting gracefully")
    with open(os.path.join(pwd, "buffer.txt"),"w") as f:
        f.write("")
    sys.exit(0)

# Function for reading lines from stdin and putting them into the queue
def read_lines(input_queue):
    while True:
        try:
            lock = FileLock(os.path.join(pwd, "buffer.lock"), timeout=5)
            with lock:
                with open(os.path.join(pwd, "buffer.txt"),"r") as f:
                    input_queue.extend([line.strip() for line in f.readlines()][-10:])
                    input_queue[:]=input_queue[-200:]
        except Exception as e:
            print(f"Error in read_lines: {e}")
        time.sleep(0.02)

def parse_stdin(line, x_vec, y_vec, z_vec, qw_vec, qx_vec, qy_vec, qz_vec):
    """
    Parses acc in m/s^2, quaternions in quaternion unit
    """
    pattern = re.compile(r'Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+), qw = (-?\d+), qx = (-?\d+), qy = (-?\d+), qz = (-?\d+)')
    match = pattern.search(line) #use search instead of match because for match the pattern has to be at the start of the line
    if match:
        x_vec.append(int(match.group(1)) / 100.0)
        y_vec.append(int(match.group(2)) / 100.0)
        z_vec.append(int(match.group(3)) / 100.0)
        qw_vec.append(int(match.group(4)) / float((1 << 14)))
        qx_vec.append(int(match.group(5)) / float((1 << 14)))
        qy_vec.append(int(match.group(6)) / float((1 << 14)))
        qz_vec.append(int(match.group(7)) / float((1 << 14)))
        return True
    return False

def parse_calib(line):
    pattern = re.compile(r'Calibration status: sys (-?\d+), gyro (-?\d+), accel (-?\d+), mag (-?\d+)')
    match = pattern.search(line) #use search instead of match because for match the pattern has to be at the start of the line
    if match:
        sys = int(match.group(1))
        gyro = int(match.group(2))
        accel = int(match.group(3))
        mag = int(match.group(4))
        return {"sys":sys, "gyro":gyro, "accel":accel, "mag":mag}
    return False

def parse_gyro(line, gx_vec, gy_vec, gz_vec):
    pattern = re.compile(r'gx = (-?\d+), gy = (-?\d+), gz = (-?\d+)')
    match = pattern.search(line) #use search instead of match because for match the pattern has to be at the start of the line
    if match:
        gx_vec.append(int(match.group(1))/16.0)
        gy_vec.append(int(match.group(2))/16.0)
        gz_vec.append(int(match.group(3))/16.0)
        return True
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

def animate(i, input_queue, x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec, yaw_vec, pitch_vec, roll_vec, gx_vec, gy_vec, gz_vec, projected_x_vec, projected_y_vec,projected_z_vec):
    """
    x_vec,y_vec,z_vec: m/s^2
    projected_x_vec, projected_y_vec,projected_z_vec : m/s^2
    yaw_vec, pitch_vec, roll_vec: radian
    gx_vec, gy_vec, gz_vec: radian/sec
    """
    #TODO: use blit; return the actual lines (with ,) ; use set_ydata; set labels before hand
    window_size = 100


    if len(input_queue)==0:
        return
    #getting the latest status line
    for line1 in reversed(input_queue):
        if status:=parse_calib(line1):
            break
    #getting the latest output line
    for line2 in reversed(input_queue):
        if parse_stdin(line2, x_vec, y_vec, z_vec, qw_vec, qx_vec, qy_vec, qz_vec):
            break
    #getting the latest output line
    for line3 in reversed(input_queue):
        if parse_gyro(line3, gx_vec, gy_vec, gz_vec):
            break

    #skip if nothing is ready yet
    if len(qw_vec)==0:
        return
    #calcualte the angles
    qw,qx,qy,qz = qw_vec[-1],qx_vec[-1],qy_vec[-1],qz_vec[-1]
    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz
    yaw = np.arctan2(2*(qx*qy + qz*qw),(sqx - sqy - sqz + sqw))
    pitch = np.arcsin(-2*(qx*qz - qy*qw)/(sqx + sqy + sqz + sqw))
    roll = np.arctan2(2*(qy*qz + qx*qw),(-sqx - sqy + sqz + sqw))

    #update the display vec
    yaw_vec.append(yaw)
    pitch_vec.append(pitch)
    roll_vec.append(roll)
    
    r = rotation_matrix_from_quaternion(qw,qx,qy,qz)
    projected_acc = r.dot(np.array([x_vec[-1],y_vec[-1],z_vec[-1]]))
    projected_x_vec.append(projected_acc[0])
    projected_y_vec.append(projected_acc[1])
    projected_z_vec.append(projected_acc[2])

    # sin
    ax1.clear()
    ax1.plot(np.array(yaw_vec)*180/math.pi, label="world_yaw")
    ax1.plot(np.array(pitch_vec)*180/math.pi, label="world_pitch")
    ax1.plot(np.array(roll_vec)*180/math.pi, label="world_roll")
    ax1.set_xlim(0, window_size)
    ax1.set_ylim(-180, 180)
    ax1.set_yticks([-180,-90,0,90,180])
    ax1.legend(loc='upper right')
    
    ax2.clear()
    ax2.plot(np.array(gx_vec), label="body_gyro_x")
    ax2.plot(np.array(gy_vec), label="body_gyro_y")
    ax2.plot(np.array(gz_vec), label="body_gyro_z")
    ax2.set_xlim(0, window_size)
    ax2.set_ylim(-2000,2000)
    ax2.legend(loc='upper right')

    ax3.clear()
    #ax3.plot(x_vec, label="acc_x")
    #ax3.plot(y_vec, label="acc_y")
    #ax3.plot(z_vec, label="acc_z")
    ax3.plot(projected_x_vec, label="world_acc_x")
    ax3.plot(projected_y_vec, label="world_acc_y")
    ax3.plot(projected_z_vec, label="world_acc_z")
    ax3.set_xlim(0, window_size)
    ax3.set_ylim(-20,20)
    ax3.legend(loc='upper right')

    fig.suptitle(status)

    #clear the vecs periodically
    if len(yaw_vec) == window_size:
        yaw_vec[:],pitch_vec[:],roll_vec[:] = [],[],[]
        x_vec[:], y_vec[:], z_vec[:] = [],[],[]
        gx_vec[:],gy_vec[:],gz_vec[:] = [],[],[]
        projected_x_vec[:],projected_y_vec[:],projected_z_vec[:] = [],[],[]


if __name__ == "__main__":
    # Set up the interrupt handler to ensure that the animation will quit when press control+C
    signal.signal(signal.SIGINT, handle_interrupt)
    # Create a multiprocessing list
    input_queue = multiprocessing.Manager().list()

    pwd = os.path.dirname(os.path.realpath(__file__))

    # if imu_ble.py print to the stdout, then as it's not consumed, when it reaches 64KB, the subprocess will hang! 
    # this behavior is the same as python3.10 imu_ble | python3.10 flive_plot.py
    # does not show in python3.10 imu_ble & python3.10 flive_plot.py
    # if imu_ble.py does not print to the stdout, then there is no risk of hanging
    ble_process = subprocess.Popen(["python3.10",os.path.join(pwd, "imu_ble.py")],stdout=subprocess.PIPE)
    read_process = multiprocessing.Process(target=read_lines, args=(input_queue,))
    read_process.start()

    t = []
    x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec,yaw_vec,pitch_vec,roll_vec, gx_vec,gy_vec,gz_vec, projected_x_vec, projected_y_vec,projected_z_vec = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    print(parse_gyro("Linear Accel X = -17, Y = 2, Z = -10, qw = 15556, qx = 1944, qy = -4743, qz = -423, gx = -5, gy = -6, gz = -10",gx_vec,gy_vec,gz_vec))

    # Create figure for plotting
    fig = plt.figure()
    fig.suptitle('test title')
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ani = FuncAnimation(fig, animate, fargs=(input_queue, x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec, yaw_vec, pitch_vec,roll_vec, gx_vec, gy_vec, gz_vec, projected_x_vec, projected_y_vec,projected_z_vec), interval=50, cache_frame_data = False, blit=False)
    plt.show()

    # Wait for the read process to finish (e.g., when Ctrl+D is pressed for stdin)
    read_process.join()