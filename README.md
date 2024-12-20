# eye-tracking-software

## 目次

1. [開発環境](#開発環境)
2. [環境構築](#環境構築)
3. [実験手順](#実験手順)
4. [トラブルシューティング](#トラブルシューティング)

## 開発環境

![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code_1.92.1-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python_3.12.5-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

| Pythonモジュール           | バージョン |
| ------------------------- | -------- |
| altgraph                  | 0.17.4   |
| beam-eye-tracker          | 1.1.1    |
| numpy                     | 2.1.2    |
| packaging                 | 24.1     |
| pefile                    | 2023.2.7 |
| pip                       | 24.2     |
| pyinstaller               | 6.11.0   |
| pyinstaller-hooks-contrib | 2024.9   |
| pywin32-ctypes            | 0.2.3    |
| setuptools                | 75.2.0   |

<p align="right">(<a href="#top">トップへ</a>)</p>

## 環境構築

執筆中...

<p align="right">(<a href="#top">トップへ</a>)</p>

## 実験手順

### 事前準備
1. Releaseページから最新版がリリースされているか確認してください。もしバージョンが更新されている場合は、zipファイルのみをダウンロードしてください。
2. zipファイルを解凍すると、実験用ソフトが利用できるようになります。
   
### 本実験の手順
1. EyewareBeamを起動してください。
2. EyewareBeamのキャリブレーションをおこなってください。
3. 実験ソフトフォルダーのdistフォルダー内にあるeyegaze.exeと問題表示ソフトを起動してください。
4. 被験者に問題表示ソフトを解き進めてもらいます。
5. 問題を最後まで回答したら、問題表示ソフトは自動的に終了します。その後、コマンドプロンプト上でCtrl+Cキーを押してeyegaze.exeソフトを終了します。
6. もう一度、eyegaze.exeと次の問題表示ソフトを起動して、次の問題に取り組んでもらってください。
7. 被験者が交代した場合は手順2.まで戻ります。

### 注意事項
- 同一被験者で続けて実験をおこなう場合は、ALT+Cでクイックキャリブレーションを実施してください。
- なるべく被験者に対して順光となる環境で実験をおこなってください。

<p align="right">(<a href="#top">トップへ</a>)</p>

## トラブルシューティング

執筆中...

<p align="right">(<a href="#top">トップへ</a>)</p>
