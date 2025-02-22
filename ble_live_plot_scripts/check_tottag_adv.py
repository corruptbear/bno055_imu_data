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


# MAIN IMU DATA LOGGING FUNCTION -----------------------------------------------------------------------------------------
from datetime import datetime
async def log_imu_data():

  while True:
      # Scan for TotTag devices for 6 seconds
      scanner = BleakScanner()
      await scanner.start()
      await asyncio.sleep(6.0)
      await scanner.stop()

      # Iterate through all discovered TotTag devices
      for device in scanner.discovered_devices:
        if device.name == 'TotTag':
          current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          # Connect to the specified TotTag and locate the ranging data service
          print(f"Found Device: {device.address} {current_time}")



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
