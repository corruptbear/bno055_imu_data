import re
import matplotlib.pyplot as plt
import numpy as np

# saved_imu_uwb.txt : two devices in hand
with open("saved_imu_uwb3.txt", "r") as f:
    log_data = f.read()
    #print(log_data[:100])

# Define a pattern for extracting X, Y, Z, qw, qx, qy data
linear_accel_pattern_1 = re.compile(
    r"AE62D6C7-4E48-FE5D-B0B8-7D09EDFC893F.*?Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+), "
    r"qw = (-?\d+), qx = (-?\d+), qy = (-?\d+), qz = (-?\d+), gx = (-?\d+), gy = (-?\d+), gz = (-?\d+)"
)

linear_accel_pattern_2 = re.compile(
    r"9A4A244A-E780-DE69-9342-F2E7207E194E.*?Linear Accel X = (-?\d+), Y = (-?\d+), Z = (-?\d+), "
    r"qw = (-?\d+), qx = (-?\d+), qy = (-?\d+), qz = (-?\d+), gx = (-?\d+), gy = (-?\d+), gz = (-?\d+)"
)

# Define a pattern for extracting range data
range_pattern_1 = re.compile(
    r"AE62D6C7-4E48-FE5D-B0B8-7D09EDFC893F.*?Ranges to 1 devices:.*?(0x[0-9A-F]+): (\d+) mm"
)

# Your log data would be read here (log_data)

#log_data_lines = log_data.split("\n")[2000:] #saved_imu_uwb1.txt
log_data_lines = log_data.split("\n")[:10000] #saved_imu_uwb3.txt
# Extract linear accel data
linear_accel_matches_1 = linear_accel_pattern_1.findall(log_data)
linear_accel_matches_2 = linear_accel_pattern_2.findall(log_data)

# Extract range data
#range_matches = range_pattern.findall(log_data)

# Output results
print("Linear Accel Data:")
t_XYZ1 = []
t_XYZ2 = []
t_ranges = []
t1_current = 0
t2_current = 0
ranges = []
X1,Y1,Z1,qw1,qx1,qy1,qz1,gx1,gy1,gz1 = [],[],[],[],[],[],[],[],[],[]
X2,Y2,Z2,qw2,qx2,qy2,qz2,gx2,gy2,gz2 = [],[],[],[],[],[],[],[],[],[]
for line in log_data_lines:
    linear_accel_matches_1 = linear_accel_pattern_1.match(line)
    if linear_accel_matches_1:
        t1_current+=0.01
        X1.append(float(linear_accel_matches_1.group(1)))
        Y1.append(float(linear_accel_matches_1.group(2)))
        Z1.append(float(linear_accel_matches_1.group(3)))
        qw1.append(float(linear_accel_matches_1.group(4)))
        qx1.append(float(linear_accel_matches_1.group(5)))
        qy1.append(float(linear_accel_matches_1.group(6)))
        qz1.append(float(linear_accel_matches_1.group(7)))
        gx1.append(float(linear_accel_matches_1.group(8)))
        gy1.append(float(linear_accel_matches_1.group(9)))
        gz1.append(float(linear_accel_matches_1.group(10)))
        t_XYZ1.append(t1_current)
    
    linear_accel_matches_2 = linear_accel_pattern_2.match(line)
    if linear_accel_matches_2:
        t2_current+=0.01
        X2.append(float(linear_accel_matches_2.group(1)))
        Y2.append(float(linear_accel_matches_2.group(2)))
        Z2.append(float(linear_accel_matches_2.group(3)))
        qw2.append(float(linear_accel_matches_2.group(4)))
        qx2.append(float(linear_accel_matches_2.group(5)))
        qy2.append(float(linear_accel_matches_2.group(6)))
        qz2.append(float(linear_accel_matches_2.group(7)))
        gx2.append(float(linear_accel_matches_2.group(8)))
        gy2.append(float(linear_accel_matches_2.group(9)))
        gz2.append(float(linear_accel_matches_2.group(10)))
        t_XYZ2.append(t2_current)
    
    range_matches_1 = range_pattern_1.match(line)
    if range_matches_1:
        ranges.append(float(range_matches_1.group(2)))
        t_ranges.append(t1_current)
    
    

    
#print(X)
#print(ranges)


# Time intervals
#t_XYZ1 = np.arange(0, len(X1) * 0.01, 0.01)  # Time for X, Y, Z at 0.01s intervals
#t_XYZ2 = np.arange(0, len(X2) * 0.01, 0.01)  # Time for X, Y, Z at 0.01s intervals
#t_ranges = np.arange(0, len(ranges) * 0.5, 0.5)  # Time for ranges at 0.5s intervals

# Create a figure and axis
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 8))

# Plot X, Y, Z on the same axis
ax1.plot(t_XYZ1, gx1, label="X", color="blue")
ax1.plot(t_XYZ1, gy1, label="Y", color="green")
ax1.plot(t_XYZ1, gz1, label="Z", color="red")
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Gyro_raw (X,Y,Z)')
ax1.legend(loc="upper left")
ax1.set_title("device 1")

# Plot X, Y, Z on the same axis
ax2.plot(t_XYZ2, gx2, label="X", color="blue")
ax2.plot(t_XYZ2, gy2, label="Y", color="green")
ax2.plot(t_XYZ2, gz2, label="Z", color="red")
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Gyro_raw (X,Y,Z)')
ax2.legend(loc="upper left")
ax2.set_title("device 2")

# Plot X, Y, Z on the same axis
ax3.plot(t_XYZ1, X1, label="X", color="blue")
ax3.plot(t_XYZ1, Y1, label="Y", color="green")
ax3.plot(t_XYZ1, Z1, label="Z", color="red")
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('linear_acc_raw (X,Y,Z)')
ax3.legend(loc="upper left")
ax3.set_title("device 1")

# Plot X, Y, Z on the same axis
ax4.plot(t_XYZ2, X2, label="X", color="blue")
ax4.plot(t_XYZ2, Y2, label="Y", color="green")
ax4.plot(t_XYZ2, Z2, label="Z", color="red")
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('linear_acc_raw (X,Y,Z)')
ax4.legend(loc="upper left")
ax4.set_title("device 2")


ax5.plot(t_ranges, ranges, label="Range", color="purple")
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Range (mm)')
ax5.legend(loc="upper right")
ax5.set_title('Range Data')


#plt.plot(t_XYZ, X)
#plt.ylabel("distance")
plt.tight_layout()
plt.show()