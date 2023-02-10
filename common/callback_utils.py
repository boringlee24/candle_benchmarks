from tensorflow import keras
from time import perf_counter
import json
import sys
from pathlib import Path
if Path('/workspace/power_monitor').is_dir():
    sys.path.append('/workspace/power_monitor')
else:
    FILE=Path(__file__).resolve()
    ROOT = FILE.parents[1]
    GITROOT=FILE.parents[2]
    sys.path.append(f'{str(GITROOT)}/power_monitor')
from carbontracker.tracker import CarbonTrackerManual

# TODO: import carbon tracker

class RecordBatch(keras.callbacks.Callback):
    def __init__(self, save_dir, model_name, start_batch=10, end_batch=2000):
        super(RecordBatch, self).__init__()
        self.batch_time = []
        self.batch_begin = 0
        self.batch_num = 1
        self.start_batch = start_batch
        self.end_batch = end_batch
        self.save_dir = save_dir
        self.model_name = model_name
        self.tracker = ''
        Path(f'{str(ROOT)}/{save_dir}').mkdir(parents=True, exist_ok=True)
    def on_train_batch_begin(self, batch, logs=None):
        if self.batch_num == self.start_batch + self.end_batch:
            overall_path = f'{str(ROOT)}/{self.save_dir}'
            self.tracker.epoch_end(f'{overall_path}/carbon_{self.model_name}')
            with open(f'{overall_path}/time_{self.model_name}.json', 'w') as f:
                json.dump(self.batch_time[self.start_batch:], f, indent=4)
            sys.exit()
        self.batch_begin = perf_counter()
    def on_train_batch_end(self, batch, logs=None):
        duration = (perf_counter() - self.batch_begin) * 1000 # ms unit
        self.batch_num += 1        
        self.batch_time.append(duration)
        if len(self.batch_time) == self.start_batch:
            self.tracker = CarbonTrackerManual(epochs=1, monitor_epochs=1, update_interval=1,
                    components='all', epochs_before_pred=1, verbose=2)
            self.tracker.tracker.pue_manual = 1
            self.tracker.intensity_updater.ci_manual = 300
            self.tracker.epoch_start()
            
