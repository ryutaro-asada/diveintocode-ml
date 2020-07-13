# 概要
### 1:ラズパイで自動レジを作成した
### 2:詳細は 'グループワーク_提案書.pdf'
### 3:使用したコードは 'raspi_register_4_14.py'
### 4:PCでのお試し用は　'raspi_register_try.py'
    <!-- 以下のライブラリ必要 -->
    import numpy as np
    from PIL import Image
    from time import sleep
    from sklearn.cluster import KMeans
    import cv2
    import os
    import matplotlib.pyplot as plt
### 5:ipynbファイルは検討した内容