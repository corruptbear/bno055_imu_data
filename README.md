
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

when time is printed at `handle_notification`, could already see repeats;
when time is printed at `data_ready_handler`, could already see repeats;
         print("%d\n",app_get_experiment_time(0));

investigating parsing errors
 - length: 7
 - the value of i decrease by 12(or multiples of 12) each time, why???? 
 - with BLE offloading, sometimes also see the same errors

investigating why UWB data does not appear in log but available as live BLE data? there is no label "r.xx" in the pkl file we load
- resolved; it's issue in `process_tottag_data`, `experimental_tottag.py`. the default details do not have label mapping

## revM-motion implementation

with `_TEST_NO_EXP_DETAILS`: every device is valid


with `_USE_DEFAULT_EXP_DETAILS`, default exp details
at reboot, the rtc is set to the compile time
```c
      experiment_details_t details = {
         .experiment_start_time = current_timestamp,
         .experiment_end_time = current_timestamp + 604800,
         .daily_start_time = 1,
         .daily_end_time = 23,
         .num_devices = 2,
         .uids = {},
         .uid_name_mappings = {}
      };
```


offloading:

you get 5MB data for 20min

```bash
# in one tab:
python3.10 segger_download.py . last_two_char_of_tottag_hex_address
# in another tab
python3.10 quick_download_trigger.py last_two_char_of_tottag_hex_address 1
```