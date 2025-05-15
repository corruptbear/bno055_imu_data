## processing data collected from phone

### extract labeled data with the button press log approach

for example, you have a file called `button_imu_logs_250507_230833.zip`

run:
```python
from extract_with_label import extract_labeled_data_from_button_interface
extract_labeled_data_from_button_interface("button_imu_logs_250507_230833.zip")
```
the zip file will be unzipped;   
`ble_imu_data_250507_230637_unit_converted.csv` and `ble_imu_data_250507_230637_unit_converted_button_labeled.csv` (unit converted data with labels from button data) will be created in the folder
`button_imu_logs_250507_230833`

### extract labeled data with video ground truth

#### case 1: video + button press log + imu data

- step 1: annotate the video
- step 2: run the annotator `python3 annotation.py`; load the raw imu data (now the annotator will convert unit automatically if not converted yet), and use the `for_marking_the_start` mask to create a special alignment file
- step 3: 
   ```python
   from extract_with_label import extract_labeled_data_from_button_interface
   extract_labeled_data_from_video(sensor_data_path="./ble_imu_data_250429_200238_unit_converted.csv", annotation_path="./20250430_030238000_iOS.aucvl")
   ```
   the labeled data will be saved to `ble_imu_data_250429_200238_unit_converted_video_labeled.csv`

### case 2: video + audio masks + imu data

- step 1: annotate the video
- step 2: run the annotator `python3 annotation.py`, load the raw imu data, and use the corresponding masks to create the alignment file
- step 3: same as with case 1


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