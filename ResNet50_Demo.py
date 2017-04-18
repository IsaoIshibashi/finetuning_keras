# -*- coding: utf-8 -*-

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

if len(sys.argv) != 2:
    print("usage: python test_vgg16.py [image file]")
    sys.exit(1)

filename = sys.argv[1]

# 学習済みのResNet50をロード
# 構造とともに学習済みの重みも読み込まれる
model = ResNet50(weights='imagenet')
model.summary()

# 引数で指定した画像ファイルを読み込む
# 224x224にリサイズされる
img = image.load_img(filename, target_size=(224, 224))

# 読み込んだPIL形式の画像をarrayに変換
x = image.img_to_array(img)

# 3次元テンソル（rows, cols, channels) を
# 4次元テンソル (samples, rows, cols, channels) に変換
# 入力画像は1枚なのでsamples=1でよい
x = np.expand_dims(x, axis=0)

# Top-5のクラスを予測する
preds = model.predict(preprocess_input(x))
results = decode_predictions(preds, top=5)[0]
for result in results:
    print(result)
