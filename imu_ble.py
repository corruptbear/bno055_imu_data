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

def data_received_callback(address, uuid, sender_characteristic, data):
    global imu_data_received_callback_count
    #print("ble data callback",address,uuid)
    if uuid == IMU_DATA_UUID:
        #this unpacked data from burst reading
        gx,gy,gz = struct.unpack('<3h',data[:6])
        qw,qx,qy,qz = struct.unpack('<4h',data[12:20])
        laccx,laccy,laccz = struct.unpack('<3h',data[20:26])
        reg_value = data[33]
        calib_mag = reg_value & 0x03
        calib_accel = (reg_value >> 2) & 0x03
        calib_gyro = (reg_value >> 4) & 0x03
        calib_sys = (reg_value >> 6) & 0x03
        print(f"{address} Calibration status: sys {calib_sys}, gyro {calib_gyro}, accel {calib_accel}, mag {calib_mag} Linear Accel X = {laccx}, Y = {laccy}, Z = {laccz}, qw = {qw}, qx = {qx}, qy = {qy}, qz = {qz}, gx = {gx}, gy = {gy}, gz = {gz}\n")

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
                f.write(f"{address} Calibration status: sys {calib_sys}, gyro {calib_gyro}, accel {calib_accel}, mag {calib_mag} Linear Accel X = {laccx}, Y = {laccy}, Z = {laccz}, qw = {qw}, qx = {qx}, qy = {qy}, qz = {qz}, gx = {gx}, gy = {gy}, gz = {gz}\n")
        #  data_file.write('{}\t{}    {}    {}\n'.format(timestamp, hex(from_eui)[2:], hex(to_eui)[2:], range_mm))
        imu_data_received_callback_count+=1
    if uuid == RANGING_DATA_UUID:
        txt_string = f"{address} Ranges to {data[0]} devices:\n"
        for i in range(data[0]):
           txt_string += '   0x%02X: %d mm\n'%(data[(3*i)+1], struct.unpack('<H', data[(3*i)+2:(3*i)+4])[0])
        print(txt_string)


async def connect_to_device(address):
    client = BleakClient(address, use_cached=False)
    data_characteristics[address]=dict()
    try:
        await client.connect()
        for service in await client.get_services():
            for characteristic in service.characteristics:

              # Open a log file, register for data notifications, and add this TotTag to the list of valid devices
              if characteristic.uuid == IMU_DATA_UUID:
                await client.start_notify(characteristic, functools.partial(data_received_callback,address,characteristic.uuid))
                tottags[address]=client
                data_characteristics[address][characteristic.uuid]=characteristic

              #if characteristic.uuid == RANGING_DATA_UUID:
              #  await client.start_notify(characteristic, functools.partial(data_received_callback,address,characteristic.uuid))
    except Exception as e:
        print('ERROR: Unable to connect to TotTag {}'.format(device.address))
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
  for i in range(len(tottags)):
    await tottags[i].stop_notify(data_characteristics[i])
    await tottags[i].disconnect()

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
