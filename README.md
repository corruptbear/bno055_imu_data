
## live visualization

`|` will stop working if `imu_ble.py` output to the terminal but the output is not cleared.
It's not a problem if `imu_ble.py` does not print anything.

Similar problem occurs if we use a subprocess to run `imu_ble.py`.


### view single IMU data via BLE
turn on the device first, then
```bash
python3.10 imu_ble.py & python3.10 flive_plot.py

python3.10 imu_ble.py | python3.10 flive_plot.py
```

or
```bash
python3l10 subprocess_flive_plot.py
```
### view paired IMU data via BLE
turn on the devices first, then
```bash
python3.10 imu_ble.py & python3.10 2flive_plot.py

python3.10 imu_ble.py | python3.10 2flive_plot.py
```