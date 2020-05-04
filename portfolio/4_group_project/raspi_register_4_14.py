#!/usr/bin/env python
#! -*- coding: utf-8 -*-

# ーーーーー 準備 ーーーーーーー
print('# ーーーーー 準備 ーーーーーーー')
import numpy as np
import picamera
from PIL import Image
from time import sleep
from sklearn.cluster import KMeans
import cv2
import os
print('cv2.__version__')
print(cv2.__version__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### -paths- ###
print('### -paths- ###')

# 背景のパス
background_path = 'back/back_ground.jpg'
# "/home/pi/self-checkout_10/back_graund.jpeg"

# テンプレート:登録商品のパス
temp_path = "pic_data"

# 写真撮影した場合のパス
photo_filename = 'take_pic.jpg'

# 精度確認テスト用のパス
# path_camera_pic = "/home/pi/self-checkout_10/temp"
# path_camera_pic = "/home/pi/self-checkout_10/picfile"

### -スケール調整-関数 ###
print('### -スケール調整-関数 ###')
def scale_to_width(img, width):
    """
    アスペクト比を固定して、幅が指定した値になるようリサイズする。
    """
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)

### -登録商品と価格 ###
print('### -登録商品と価格 ###')
money = {'Rooibos_tea':110, 'CC_lemon':120, 'Aquarius':130, 'Carbonated_water':140, 'Jasmin_tea':150}
# Aquarius.jpg  CC_lemon.jpg  Carbonated_water.jpg  Jasmin_tea.jpg  Jogier_coffee.jpg  Pepsi_zero.jpg  Rooibos_tea.jpg

### -background-準備 ###
print('### -background-準備 ###')
background_image = cv2.imread(background_path)

#ラズパイカメラに形状を合わせる(使う場合は)
# background_image = cv2.resize(background_image, dsize= (1920, 1080), interpolation = cv2.INTER_AREA)

# 色変換
background_image= cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# スケールダウン　必要に応じて
# background_image = scale_to_width(background_image, 224)

# 画像にぼかしをかける（誤検出防止のためのノイズ除去）:cv2.GaussianBlur(入力画像,(畳み込むサイズ),標準偏差）
background_image = cv2.GaussianBlur(background_image, (5, 5), 0)

# 背景画像確認用に出力
plt.imshow(background_image)
plt.show()
plt.savefig("background_BLUR.jpg")

print('＝＝＝＝＝＝＝ 物体検出　関数 ＝＝＝＝＝＝')
### bounding_boxの範囲決定-関数 ###
def get_x_y_limit(Y, X, result, cluster):
    NO = np.where(result==cluster)
    x_max = np.max(X[NO])
    x_min = np.min(X[NO])
    y_max = np.max(Y[NO])
    y_min = np.min(Y[NO])
    x_max = int(x_max)
    x_min = int(x_min)
    y_max = int(y_max)
    y_min = int(y_min)
    return x_min, y_min, x_max, y_max

### bounding_boxの生成と切り取り-関数 ###
def bounding_box(img, x_min, y_min, x_max, y_max):

    # 長方形を生成cv2.rectangle(イメージデータ、（左上の座標)、（右下の座標）、（BGR色指定）、（線の太さ））
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)

    # bounding_box部分のみ切り取り
    truncated = img[y_min:y_max, x_min:x_max]

    # imgはbox付きの元画像、truncatedは切り取った画像
    return img, truncated

### -main-物体検出-関数 ###
def objection_detect(background_image, product_image):

    # 画像にぼかしをかける（誤検出防止のためのノイズ除去）:cv2.GaussianBlur(入力画像,(畳み込むサイズ),標準偏差）
    product_image_Blured = cv2.GaussianBlur(product_image, (5, 5), 0)

    # 背景差分するためのmaskモデル
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    # 背景の投入
    fgmask = fgbg.apply(np.uint8(background_image))

    # 撮影画像の投入
    fgmask = fgbg.apply(np.uint8(product_image_Blured))

    # 差がある部分の座標を抽出
    Y, X = np.where(fgmask > 200)

    if len(Y) == 0 or len(X) == 0:
        return X, Y

    # Kmansによる背景と入力画像との違いをクラスタ分類
    y = KMeans(n_clusters=1, random_state=0).fit_predict(np.array([X,Y]).T)

    # 上記で定義したget_x_y_limit関数による、bounding_boxの範囲の抽出
    x_min, y_min, x_max, y_max = get_x_y_limit(Y, X, y, 0)

    ### bounding_boxつき画像と切り取り画像を返す ###
    return bounding_box(product_image, x_min, y_min, x_max, y_max)

# ＝＝＝＝＝＝＝ 物体検出 終わり ＝＝＝＝＝＝

### -登録商品画像５個-準備 ###
print('### -登録商品画像５個-準備 ###')
temp_files = os.listdir(temp_path)
print(temp_files)
temp_data = []
count = 0
for data in temp_files:

    # 登録写真データ読み込み
    temp = cv2.imread(temp_path + "/" + data)

    # 登録写真データの形状が撮影写真データと違う場合
    # temp = cv2.resize(temp, dsize= (1920, 1080), interpolation = cv2.INTER_AREA)

    temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)

    # 必要に応じて
    # temp = scale_to_width(temp, 224)

    box_added_pic, temp = objection_detect(background_image, temp)

    # 物体検出あとの切り取り画像の保存（確認テスト用）
    plt.imshow(temp)
    plt.show()
    plt.savefig(str(count)+"templates.jpg")

    count += 1
    print('テンプレ切り取りあと')
    print(data)
    print(temp.shape)
    temp_data.append(temp)

### -Test用カメラ撮影画像-準備 ###
# print('### - 精度確認テスト用画像-準備 ###')
# camera_files = os.listdir(path_camera_pic)
# camera_files = [files for files in camera_files]
# print(camera_files)

### -カメラ撮影-関数 ###
print('### -カメラ撮影-関数 ###')
def shutter():
    """
    デフォルト解像度　-w 1920 -h 1080
    カメラで写真を撮って'/tmp/data.jpg'に保存する。
    撮影のプレビュー見せる。
    """
    # pi camera 用のライブラリーを使用して、画像を取得
    with picamera.PiCamera() as camera:
        camera.start_preview()
        sleep(1.000)
        camera.capture(photo_filename)
        camera.stop_preview()



print('＝＝＝＝＝＝＝ 画像分類 関数　＝＝＝＝＝＝')
### -main-画像分類-関数 ###
def classfiler(temp_files, temp_data, img, param=0.59, num=6):
    """
    テンプレート画像と撮った写真がマッチしているかどうか。
    マッチしている場合はマッチしたテンプレート名を返す。
    マッチしてない場合は何も返さない。
    詳細:
    SIFTを用いた画像分類
    パラメーターとして以下を調整
    param：最も近い点と2番目に近い点の距離の比率。
    num：マッチした特徴点の数の下限
    """
    # 画像マッチング テンプレートの写真とマッチしているかどうかの判断
    for temp, temp_file in zip(temp_data, temp_files):

        #特徴抽出機の生成
        # detector = cv2.AgastFeatureDetector_create()  #エラー
        # detector = cv2.FastFeatureDetector_create()  #エラー
        # detector = cv2.MSER_create()  #エラー
        # detector = cv2.AKAZE_create()  #エラー
        # detector = cv2.BRISK_create()  #エラー
        # detector = cv2.KAZE_create()    #暗くても当たりやすいが遅い
        # detector = cv2.ORB_create()   #途中でエラー
        # detector = cv2.SimpleBlobDetector_create()  #エラー
        detector = cv2.xfeatures2d.SIFT_create()

        #kpは特徴的な点の位置 destは特徴を現すベクトル
        kp1, des1 = detector.detectAndCompute(temp, None)
        kp2, des2 = detector.detectAndCompute(img, None)

        #特徴点の比較機
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        #割合試験を適用(２番目との差を param)
        good = []
        match_param = param
        for m,n in matches:
            if m.distance < match_param*n.distance:
                good.append([m])

        # good特徴点 num 個以上でマッチしたとする
        if len(good)>num:
            name = temp_file.strip(".jpg'")#.jpgとる
            return name

print('ーーーーー 準備 終わり ーーーーーーー')

print('========= main_part ===========')
if __name__ == '__main__':
    while True:
        money_sum = 0 

        # 何らかのキー操作があるまで待機（何でもいいが、とりあえず「Enter」を押すように伝える）
        key = input('商品をスキャンする場合は「Enter」を押して下さい')
        while True:
            # 実験用コード
            # for files in camera_files:
            # print(files)
            # img = cv2.imread(path_camera_pic + "/" + files)

            # 写真撮影と撮影画像読み込み
            shutter()
            img = cv2.imread(photo_filename)

            # 画像をBGRカラーに変換
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # スケールダウン　必要に応じて
            # img = scale_to_width(img, 224)

            # 物体検出
            img, truncated = objection_detect(background_image, img)

            if len(img) == 0 or len(truncated) == 0:
                print('何も入ってません')

                key = input('続けて商品をスキャンする場合は「y + Enter」,会計する場合は「Enter」を押して下さい')
                if key != 'y':
                    print("合計:{}円".format(money_sum))
                    break

                continue

            # 物体検出あとの切り取り画像の保存（確認用）
            plt.imshow(truncated)
            plt.show()
            plt.savefig("pic_after_OD.jpg")

            # 分類、登録商品とそれ以外で分けてる
            name = classfiler(temp_files, temp_data, truncated, param=0.59, num=6)

            # 分類関数から名前が出力されない場合は未登録を連絡。
            if not name:
                print("登録商品ではありません")

            # その他　マッチした商品名を伝えて、今の小計金額を連絡。
            else:
                print(name)
                money_sum += money[name]
                print("小計",money_sum)

            # y以外の入力があった場合は合計金額を伝えて、ループをブレークし、money_sum = 0に戻す
            key = input('続けて商品をスキャンする場合は「y + Enter」,会計する場合は「Enter」を押して下さい')
            if key != 'y':
                print("合計:{}円".format(money_sum))
                break