
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

## processing data


1. convert .pkl to .csv using `load_imu_data.py`

2. annotate data using `annotator.py`
the script loads the .csv file and set the starting timestamp to 0
save the `_alignment.yaml`

3. extract labels using `extract_with_label.py`
given the path of the .csv file, generate labeled data using the `_alignment.yaml`

## TODO

investigating why the IMU timestamps repeat?
- print out timestamp at data read interrupt time, check swo output
- print out timestamp at write storage time, check swo output
ultimately, in the training data, we assume 100 Hz data, timestamps are not used


investigating why UWB data does not appear in log but available as live BLE data? there is no label "r.xx" in the pkl file we load
- perhaps because there is no label in the test mode???