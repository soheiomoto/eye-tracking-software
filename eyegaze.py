"""
  eyegaze.py:
  - version 1.0.0

  Dependencies:
  - Python 3.12.5
  - beam-eye-tracker
  - NumPy
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

#SDKがあるディレクトリにPATHを通す
path = os.getcwd()
path_name = path + "\\BeamSDK-Windows64-1.1.0\\API\\python"
sys.path.append(path_name)

from eyeware.client import TrackerClient

tracker = TrackerClient()

now = datetime.now()
time_str = now.strftime("%H%M%S")
name = 'tracker_record_'+time_str+'.txt'

def gaze_tracker():
    with open(name, 'w') as f:
        while True:
            if tracker.connected:
                record_now = datetime.now()
                record_time = record_now.strftime("%H%M%S")
                
                head_pose = tracker.get_head_pose_info()
                head_is_lost = head_pose.is_lost
                if not head_is_lost:
                    rot = head_pose.transform.rotation
                    tr = head_pose.transform.translation

                print(record_time, file=f)
                print("  * Gaze on Screen:",file=f)
                screen_gaze = tracker.get_screen_gaze_info()
                screen_gaze_is_lost = screen_gaze.is_lost
                print("      - Lost track:       ", screen_gaze_is_lost,file=f)
                if not screen_gaze_is_lost:
                    print("      - Coordinates:       <x=%5.3f px,   y=%5.3f px>" % (screen_gaze.x, screen_gaze.y),file=f)
                
                #30hzでデータを取得
                time.sleep(1 / 30)
            else:
                #APIとの通信が失われている場合は1秒おきにエラー文をプリント
                MESSAGE_PERIOD_IN_SECONDS = 1
                time.sleep(MESSAGE_PERIOD_IN_SECONDS - time.monotonic() % MESSAGE_PERIOD_IN_SECONDS)
                print("No connection with tracker server")

if __name__ == "__main__":
    gaze_tracker()