#!/usr/bin/env python

#-------------------this only works with flive_plot.py---------------------
# the normal live_plot.py for some reasong is very laggy working through pipe with BLE
#python3.10 imu_ble.py | python3.10 flive_plot.py
#or(for viewing the output of this script)
#python3.10 imu_ble.py & python3.10 flive_plot.py

# PYTHON INCLUSIONS ---------------------------------------------------------------------------------------------------

import asyncio
import functools
import struct
import sys
from bleak import BleakClient, BleakScanner
from datetime import datetime
import os
from filelock import Timeout, FileLock
import signal
import traceback

CONNECTION_TIMEOUT=6

# Helper function to handle Ctrl+C
def handle_interrupt(sig, frame):
    print("\nExiting gracefully...")
    with open(os.path.join(pwd, "buffer.txt"),"w") as f:
        f.write("")
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    loop.stop()

# CONSTANTS AND DEFINITIONS -------------------------------------------------------------------------------------------

IMU_DATA_UUID =     'd68c3158-a23f-ee90-0c45-5231395e5d2e'
RANGING_DATA_UUID = 'd68c3156-a23f-ee90-0c45-5231395e5d2e'
BATCH_SIZE = 2

# STATE VARIABLES -----------------------------------------------------------------------------------------------------

filename_base = datetime.now().strftime('ranging_data_%Y-%m-%d_%H-%M-%S_')
data_characteristics = dict()
ranging_files = []
tottags = dict()

pwd = os.path.dirname(os.path.realpath(__file__))
#create the lock file if it does not exist
if not os.path.exists(os.path.join(pwd, "buffer.lock")):
    # Create the file
    with open(os.path.join(pwd, "buffer.lock"), "w") as f:
        f.write("")
with open(os.path.join(pwd, "buffer.txt"),"w") as f:
    f.write("")

# HELPER FUNCTIONS AND CALLBACKS --------------------------------------------------------------------------------------
def unpack_bno055_burst(data):
    """
    the data include the 4 byte timestamp and the chunk of data starting from acc
    """
    timestamp = struct.unpack('<I', data[:4])[0]
    accx,accy,accz = struct.unpack('<3h',data[4:10])
    magx,magy,magz = struct.unpack('<3h',data[10:16])
    gx,gy,gz = struct.unpack('<3h',data[16:22])
    qw,qx,qy,qz = struct.unpack('<4h',data[28:36])
    laccx,laccy,laccz = struct.unpack('<3h',data[36:42])
    reg_value = data[49]
    calib_mag = reg_value & 0x03
    calib_accel = (reg_value >> 2) & 0x03
    calib_gyro = (reg_value >> 4) & 0x03
    calib_sys = (reg_value >> 6) & 0x03
    return timestamp,accx,accy,accz,gx,gy,gz,qw,qx,qy,qz,laccx,laccy,laccz,calib_mag,calib_accel,calib_gyro,calib_sys,magx,magy,magz


def unpack_bno055_full_imu(data):
    print(len(data))
    unpacked = []
    #timestamp, gx,gy,gz,qw,qx,qy,qz,laccx,laccy,laccz,calib_mag,calib_accel,calib_gyro,calib_sys = unpack_bno055_burst(data)
    unpacked.append(unpack_bno055_burst(data))
    unpacked.append(unpack_bno055_burst(data[52:]))
    #timestamp2, gx2,gy2,gz2,qw2,qx2,qy2,qz2,laccx2,laccy2,laccz2,calib_mag2,calib_accel2,calib_gyro2,calib_sys2 = unpack_bno055_burst(data[52:])

    #return timestamp,gx,gy,gz,qw,qx,qy,qz,laccx,laccy,laccz,calib_mag,calib_accel,calib_gyro,calib_sys
    return unpacked

def data_received_callback(address, uuid, sender_characteristic, data):
    global imu_data_received_callback_count
    #print("ble data callback",address,uuid)
    if uuid == IMU_DATA_UUID:
        #this unpacked data from burst reading
        unpacked = unpack_bno055_full_imu(data)
        for i in range(BATCH_SIZE):
            timestamp,accx,accy,accz,gx,gy,gz,qw,qx,qy,qz,laccx,laccy,laccz,calib_mag,calib_accel,calib_gyro,calib_sys,magx,magy,magz = unpacked[i]
            print(f"{timestamp} {address} Calibration status: sys {calib_sys}, gyro {calib_gyro}, accel {calib_accel}, mag {calib_mag} Raw Accel X = {accx}, Y = {accy}, Z = {accz}, Linear Accel X = {laccx}, Y = {laccy}, Z = {laccz}, qw = {qw}, qx = {qx}, qy = {qy}, qz = {qz}, gx = {gx}, gy = {gy}, gz = {gz}, magx = {magx}, magy = {magy}, magz = {magz}\n")

        lock = FileLock(os.path.join(pwd, "buffer.lock"), timeout=5)
        with lock:
            #wipe the buffer file so that it does not grow too huge
            if imu_data_received_callback_count>=200:
                #reset the counter
                imu_data_received_callback_count=0
                with open(os.path.join(pwd, "buffer.txt"),"w") as f:
                    f.write("")
            #append data to the buffer file
            with open(os.path.join(pwd, "buffer.txt"),"a") as f:
                #f.write(f"Calibration status: sys {calib_sys}, gyro {calib_gyro}, accel {calib_accel}, mag {calib_mag}\n")
                for i in range(BATCH_SIZE):
                    timestamp,accx,accy,accz,gx,gy,gz,qw,qx,qy,qz,laccx,laccy,laccz,calib_mag,calib_accel,calib_gyro,calib_sys,magx,magy,magz = unpacked[i]
                    f.write(f"{address} Calibration status: sys {calib_sys}, gyro {calib_gyro}, accel {calib_accel}, mag {calib_mag} Raw Accel X = {accx}, Y = {accy}, Z = {accz}, Linear Accel X = {laccx}, Y = {laccy}, Z = {laccz}, qw = {qw}, qx = {qx}, qy = {qy}, qz = {qz}, gx = {gx}, gy = {gy}, gz = {gz}, magx = {magx}, magy = {magy}, magz = {magz}\n")
        #  data_file.write('{}\t{}    {}    {}\n'.format(timestamp, hex(from_eui)[2:], hex(to_eui)[2:], range_mm))
        imu_data_received_callback_count+=1
    if uuid == RANGING_DATA_UUID:
        txt_string = f"{address} Ranges to {data[0]} devices:"
        for i in range(data[0]):
           txt_string += '   0x%02X: %d mm\n'%(data[(3*i)+1], struct.unpack('<H', data[(3*i)+2:(3*i)+4])[0])
        print(txt_string)


async def connect_to_device(address):
    client = BleakClient(address, use_cached=False)
    data_characteristics[address]=[]
    try:
        await client.connect()
        imu_characteristic_found = False
        ranging_characteristic_found = False
        if client.is_connected:
            for service in client.services:
                for characteristic in service.characteristics:
                    # Open a log file, register for data notifications, and add this TotTag to the list of valid devices
                    if characteristic.uuid == IMU_DATA_UUID:
                        #print(service.uuid)
                        await client.start_notify(IMU_DATA_UUID, functools.partial(data_received_callback,address,IMU_DATA_UUID))
                        tottags[address]=client
                        data_characteristics[address].append(IMU_DATA_UUID)
                        imu_characteristic_found = True

                    if characteristic.uuid == RANGING_DATA_UUID:
                        await client.start_notify(RANGING_DATA_UUID, functools.partial(data_received_callback,address,RANGING_DATA_UUID))
                        data_characteristics[address].append(RANGING_DATA_UUID)
                        ranging_characteristic_found = True
            print(f"imu characteristic found: {imu_characteristic_found}")
            print(f"ranging characteristic found: {ranging_characteristic_found}")


    except Exception as e:
        print('ERROR: Unable to connect to TotTag {}'.format(client.address))
        traceback.print_exc()
        await client.disconnect()

# MAIN IMU DATA LOGGING FUNCTION -----------------------------------------------------------------------------------------

async def log_imu_data():

  # Scan for TotTag devices for 6 seconds
  scanner = BleakScanner()
  await scanner.start()
  await asyncio.sleep(6.0)
  await scanner.stop()

  # Iterate through all discovered TotTag devices
  for device in scanner.discovered_devices:
    if device.name == 'TotTag':
      # Connect to the specified TotTag and locate the ranging data service
      print('Found Device: {}'.format(device.address))
      await asyncio.wait_for(connect_to_device(device.address), timeout=CONNECTION_TIMEOUT)

  # Wait forever while ranging data is being logged
  while True:
      #print(data_characteristics)
      for device_address in tottags:
          client = tottags[device_address]
          if not client.is_connected:
              print(f"{device.address} client disconected!")
              #re-connect
              await connect_to_device(device_address)

      await asyncio.sleep(1)

  # Disconnect from all TotTag devices
  for address in tottags:
      for i in range(len(data_characteristics[address])):
          await tottags[address].stop_notify(data_characteristics[address][i])
      await tottags[address].disconnect()

# TOP-LEVEL FUNCTIONALITY ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)
    imu_data_received_callback_count = 0

    print('\nSearching 6 seconds for TotTags...\n')
    loop = asyncio.get_event_loop()

    # Listen forever for TotTag ranges
    try:
      loop.run_until_complete(log_imu_data())

    # Gracefully close all log files
    finally:
      loop.close()
