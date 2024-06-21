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

def parse_addr(addr, line):
    pattern = re.compile(r'[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}', re.IGNORECASE)
    if pattern.search(line):
        return True
    return False

def parse_stdin(addr, line):
    """
    Parses acc in m/s^2, quaternions in quaternion unit
    """
    if addr not in line:
        return False
    pattern = re.compile(r'Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+), qw = (-?\d+), qx = (-?\d+), qy = (-?\d+), qz = (-?\d+)')
    match = pattern.search(line) #use search instead of match because for match the pattern has to be at the start of the line
    if match:
        data[addr]["x_vec"].append(int(match.group(1)) / 100.0)
        data[addr]["y_vec"].append(int(match.group(2)) / 100.0)
        data[addr]["z_vec"].append(int(match.group(3)) / 100.0)
        data[addr]["qw_vec"].append(int(match.group(4)) / float((1 << 14)))
        data[addr]["qx_vec"].append(int(match.group(5)) / float((1 << 14)))
        data[addr]["qy_vec"].append(int(match.group(6)) / float((1 << 14)))
        data[addr]["qz_vec"].append(int(match.group(7)) / float((1 << 14)))
        return True
    return False

def parse_calib(addr, line):
    if addr not in line:
        return False
    pattern = re.compile(r'Calibration status: sys (-?\d+), gyro (-?\d+), accel (-?\d+), mag (-?\d+)')
    match = pattern.search(line) #use search instead of match because for match the pattern has to be at the start of the line
    if match:
        sys = int(match.group(1))
        gyro = int(match.group(2))
        accel = int(match.group(3))
        mag = int(match.group(4))
        return {"sys":sys, "gyro":gyro, "accel":accel, "mag":mag}
    return False

def parse_gyro(addr, line):
    if addr not in line:
        return False
    pattern = re.compile(r'gx = (-?\d+), gy = (-?\d+), gz = (-?\d+)')
    match = pattern.search(line) #use search instead of match because for match the pattern has to be at the start of the line
    if match:
        data[addr]["gx_vec"].append(int(match.group(1))/16.0)
        data[addr]["gy_vec"].append(int(match.group(2))/16.0)
        data[addr]["gz_vec"].append(int(match.group(3))/16.0)
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

def animate(i, addr, input_queue):
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
    for line in reversed(input_queue):
        if status:=parse_calib(addr, line):
            print(addr, line)
            break
    #getting the latest output line
    for line in reversed(input_queue):
        if parse_stdin(addr, line):
            #print(addr, line)
            break
    #getting the latest output line
    for line in reversed(input_queue):
        if parse_gyro(addr, line):
            #print(addr, line)
            break

    #skip if nothing is ready yet
    if len(data[addr]["qw_vec"])==0 or len(data[addr]["x_vec"])==0:
        return
    #calcualte the angles
    qw,qx,qy,qz = data[addr]["qw_vec"][-1],data[addr]["qx_vec"][-1],data[addr]["qy_vec"][-1],data[addr]["qz_vec"][-1]
    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz
    yaw = np.arctan2(2*(qx*qy + qz*qw),(sqx - sqy - sqz + sqw))
    pitch = np.arcsin(-2*(qx*qz - qy*qw)/(sqx + sqy + sqz + sqw))
    roll = np.arctan2(2*(qy*qz + qx*qw),(-sqx - sqy + sqz + sqw))

    #update the display vec
    data[addr]['yaw_vec'].append(yaw)
    data[addr]['pitch_vec'].append(pitch)
    data[addr]['roll_vec'].append(roll)

    #projected acc is already rotated from the body frame to world frame
    r = rotation_matrix_from_quaternion(qw,qx,qy,qz)
    projected_acc = r.dot(np.array([data[addr]["x_vec"][-1],data[addr]["y_vec"][-1],data[addr]["z_vec"][-1]]))
    data[addr]['projected_x_vec'].append(projected_acc[0])
    data[addr]['projected_y_vec'].append(projected_acc[1])
    data[addr]['projected_z_vec'].append(projected_acc[2])

    ax1,ax2,ax3 = data[addr]['ax1'],data[addr]['ax2'],data[addr]['ax3']
    # sin
    ax1.clear()
    ax1.axhline(y=-90, color='gray', linestyle='--', linewidth=0.5)
    ax1.axhline(y=-45, color='gray', linestyle='--', linewidth=0.5)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axhline(y=45, color='gray', linestyle=':', linewidth=0.5)
    ax1.axhline(y=90, color='gray', linestyle=':', linewidth=0.5)
    ax1.plot(np.array(data[addr]['yaw_vec'])*180/math.pi, label="world_yaw")
    ax1.plot(np.array(data[addr]['pitch_vec'])*180/math.pi, label="world_pitch")
    ax1.plot(np.array(data[addr]['roll_vec'])*180/math.pi, label="world_roll")
    ax1.set_xlim(0, window_size)
    ax1.set_ylim(-180, 180)
    ax1.set_yticks([-180,-90,0,90,180])
    ax1.legend(loc='upper right')
    
    ax2.clear()
    ax2.plot(np.array(data[addr]["gx_vec"]), label="body_gyro_x")
    ax2.plot(np.array(data[addr]["gy_vec"]), label="body_gyro_y")
    ax2.plot(np.array(data[addr]["gz_vec"]), label="body_gyro_z")
    ax2.set_xlim(0, window_size)
    ax2.set_ylim(-2000,2000)
    ax2.legend(loc='upper right')

    ax3.clear()
    #ax3.plot(x_vec, label="acc_x")
    #ax3.plot(y_vec, label="acc_y")
    #ax3.plot(z_vec, label="acc_z")
    ax3.plot(data[addr]['projected_x_vec'], label="world_acc_x")
    ax3.plot(data[addr]['projected_y_vec'], label="world_acc_y")
    ax3.plot(data[addr]['projected_z_vec'], label="world_acc_z")
    ax3.set_xlim(0, window_size)
    ax3.set_ylim(-20,20)
    ax3.legend(loc='upper right')

    data[addr]['fig'].suptitle(f"{addr}\n{status}")

    #clear the vecs periodically
    if len(data[addr]["yaw_vec"]) == window_size:
        data[addr]["yaw_vec"][:], data[addr]["pitch_vec"][:], data[addr]["roll_vec"][:] = [],[],[]
        data[addr]["x_vec"][:], data[addr]["y_vec"][:], data[addr]["z_vec"][:] = [],[],[]
        data[addr]["gx_vec"][:], data[addr]["gy_vec"][:], data[addr]["gz_vec"][:] = [],[],[]
        data[addr]["projected_x_vec"][:], data[addr]["projected_y_vec"][:], data[addr]["projected_z_vec"][:] = [],[],[]

def add_animation(addr):
    t = []
    #x_vec,y_vec,z_vec,qw_vec,qx_vec,qy_vec,qz_vec,yaw_vec,pitch_vec,roll_vec, gx_vec,gy_vec,gz_vec, projected_x_vec, projected_y_vec,projected_z_vec = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    # Create figure for plotting
    fig = plt.figure()
    fig.suptitle('test title')
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    data[addr]['ax1'],data[addr]['ax2'],data[addr]['ax3']=ax1,ax2,ax3
    data[addr]['fig']=fig

    ani = FuncAnimation(fig, animate, fargs=(addr,input_queue), interval=50, cache_frame_data = False, blit=False)
    return ani

if __name__ == "__main__":
    # Set up the interrupt handler to ensure that the animation will quit when press control+C
    signal.signal(signal.SIGINT, handle_interrupt)
    # Create a multiprocessing list
    input_queue = multiprocessing.Manager().list()

    # Create and start the process to read lines
    # Multiprocessing is needed to only get the latest stuff from stdin
    read_process = multiprocessing.Process(target=read_lines, args=(input_queue,))
    read_process.start()

    anis=[]
    data=dict()

    while True:
        #loop until 2 devices are connected
        with open(os.path.join(pwd, "buffer.txt"),"r") as f:
            lines = f.readlines()
            pattern = re.compile(r'[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}', re.IGNORECASE)
            matches = [pattern.search(line) for line in lines]
            addrs = set([match.group() for match in matches if match])
            if len(addrs)==2:
                for addr in addrs:
                    data[addr]={'x_vec': [], 'y_vec': [], 'z_vec': [], 'qw_vec': [], 'qx_vec': [], 'qy_vec': [], 'qz_vec': [], 'yaw_vec': [], 'pitch_vec': [], 'roll_vec': [],'gx_vec': [], 'gy_vec': [], 'gz_vec': [], 'projected_x_vec': [], 'projected_y_vec': [], 'projected_z_vec': []}
                break

    for addr in addrs:
        anis.append(add_animation(addr))
    #anis.append(add_animation())
    #anis.append(add_animation())
    #add_animation()
    plt.show()

    # Wait for the read process to finish (e.g., when Ctrl+D is pressed for stdin)
    read_process.join()