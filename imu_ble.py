#!/usr/bin/env python

#-------------------this only works with flive_plot.py---------------------
# the normal live_plot.py for some reasong is very laggy working through pipe with BLE
#python3.10 imu_ble.py | python3.10 flive_plot.py

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
# Helper function to handle Ctrl+C
def handle_interrupt(sig, frame):
    print("\nExiting gracefully...")
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    loop.stop()

# CONSTANTS AND DEFINITIONS -------------------------------------------------------------------------------------------

IMU_DATA_UUID = 'd68c3158-a23f-ee90-0c45-5231395e5d2e'


# STATE VARIABLES -----------------------------------------------------------------------------------------------------

filename_base = datetime.now().strftime('ranging_data_%Y-%m-%d_%H-%M-%S_')
data_characteristics = []
ranging_files = []
tottags = []
pwd = os.path.dirname(os.path.realpath(__file__))

# HELPER FUNCTIONS AND CALLBACKS --------------------------------------------------------------------------------------

def data_received_callback(sender_characteristic, data):
    print("ble data callback")
    if not hasattr(data_received_callback, "count"):
        data_received_callback.count = 0  # Initialize the static variable
    data_received_callback.count+=1

    #this unpacked data from burst reading
    gx,gy,gz = struct.unpack('<3h',data[:6])
    qw,qx,qy,qz = struct.unpack('<4h',data[12:20])
    laccx,laccy,laccz = struct.unpack('<3h',data[20:26])
    reg_value = data[33]
    calib_mag = reg_value & 0x03
    calib_accel = (reg_value >> 2) & 0x03
    calib_gyro = (reg_value >> 4) & 0x03
    calib_sys = (reg_value >> 6) & 0x03

    lock = FileLock(os.path.join(pwd, "temp.lock"), timeout=5)
    with lock:
        #wipe the buffer file so that it does not grow too huge
        if data_received_callback.count==100:
            with open(os.path.join(pwd, "temp.txt"),"w") as f:
                f.write("")
            #reset the counter
            data_received_callback.count==0
        #append data to the buffer file
        with open(os.path.join(pwd, "temp.txt"),"a") as f:
            f.write(f"Calibration status: sys {calib_sys}, gyro {calib_gyro}, accel {calib_accel}, mag {calib_mag}\n")
            f.write(f"Linear Accel X = {laccx}, Y = {laccy}, Z = {laccz}, qw = {qw}, qx = {qx}, qy = {qy}, qz = {qz}, gx = {gx}, gy = {gy}, gz = {gz}\n")
   #  data_file.write('{}\t{}    {}    {}\n'.format(timestamp, hex(from_eui)[2:], hex(to_eui)[2:], range_mm))


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
      client = BleakClient(device, use_cached=False)
      try:
        await client.connect()
        for service in await client.get_services():
          for characteristic in service.characteristics:

            # Open a log file, register for data notifications, and add this TotTag to the list of valid devices
            if characteristic.uuid == IMU_DATA_UUID:
              #try:
              #  file = open(filename_base + client.address.replace(':', '') + '.data', 'w')
              #  file.write('Timestamp\tFrom  To    Distance (mm)\n')
              #except Exception as e:
              #  print(e)
              #  print('ERROR: Unable to create a ranging data log file')
              #  sys.exit('Unable to create a ranging data log file: Cannot continue!')
              await client.start_notify(characteristic, functools.partial(data_received_callback))
              data_characteristics.append(characteristic)
              tottags.append([client,device])

      except Exception as e:
        print('ERROR: Unable to connect to TotTag {}'.format(device.address))
        await client.disconnect()

  # Wait forever while ranging data is being logged
  while True:
      for client, device in tottags:
          if not client.is_connected:
              print(f"{device.address} client disconected!")
      await asyncio.sleep(1)

  # Disconnect from all TotTag devices
  for i in range(len(tottags)):
    await tottags[i].stop_notify(data_characteristics[i])
    await tottags[i].disconnect()

# TOP-LEVEL FUNCTIONALITY ---------------------------------------------------------------------------------------------

signal.signal(signal.SIGINT, handle_interrupt)

print('\nSearching 6 seconds for TotTags...\n')
loop = asyncio.get_event_loop()

# Listen forever for TotTag ranges
try:
  loop.run_until_complete(log_imu_data())

# Gracefully close all log files
finally:
  loop.close()