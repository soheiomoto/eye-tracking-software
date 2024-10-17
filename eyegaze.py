"""
  eyegaze.py:
  - version 1.1.1

  Dependencies:
  - beam-eye-tracker          1.1.1
  - numpy                     2.1.2
  - pyinstaller               6.10.0

  hiddenimports:
  - numpy.core.multiarray
"""

# プログラムに必要なモジュールのインポート
import csv
import time
import numpy as np
from datetime import datetime
from eyeware.client import TrackerClient

tracker = TrackerClient()

# 時間記録・ファイル名用の変数設定
now = datetime.now()
time_str = now.strftime("%H%M%S")
name = 'tracker_record_'+time_str+'.csv'

def gaze_tracker():
    with open(name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # CSVファイル内のヘッダー行の作成
        csv_writer.writerow(["Record Time", "Gaze Lost", "X Coordinate", "Y Coordinate"])
        
        while True:
            if tracker.connected:
                record_now = datetime.now()
                record_time = record_now.strftime("%H%M%S%f")
                status_time = record_now.strftime("%H%M%S")
                
                head_pose = tracker.get_head_pose_info()
                head_is_lost = head_pose.is_lost
                
                # 座標情報初期化
                screen_gaze_x = None
                screen_gaze_y = None
                
                if not head_is_lost:
                    rot = head_pose.transform.rotation
                    tr = head_pose.transform.translation

                screen_gaze = tracker.get_screen_gaze_info()
                screen_gaze_is_lost = screen_gaze.is_lost
                
                if screen_gaze_is_lost:
                    gaze_status = "Lost"
                    print('lost')
                    screen_gaze_x = None
                    screen_gaze_y = None
                else:
                    gaze_status = "Visible"
                    print('visible')
                    screen_gaze_x = screen_gaze.x
                    screen_gaze_y = screen_gaze.y
                
                # データ行の書き込み
                csv_writer.writerow([record_time, gaze_status, screen_gaze_x, screen_gaze_y])
                
                # 60hzでデータを収集
                time.sleep(1 / 60)
            else:
                # 接続が切れている場合は毎秒エラー文を表示
                MESSAGE_PERIOD_IN_SECONDS = 1
                time.sleep(MESSAGE_PERIOD_IN_SECONDS - time.monotonic() % MESSAGE_PERIOD_IN_SECONDS)
                print("No connection with tracker server")

if __name__ == "__main__":
    gaze_tracker()