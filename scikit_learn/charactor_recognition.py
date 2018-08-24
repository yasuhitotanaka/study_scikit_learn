# coding: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数学データの読み込み
digits = datasets.load_digits()

# データの形式を確認
# print digits.data
# print digits.data.shape

# データ数
n = len(digits.data)

# # 画像と正解値の表示
# images = digits.images
# labels = digits.target
# for i in range(10):
#     # 縦2、横5つの画像を表示する
#     plt.subplot(2, 5, i + 1)
#     # グレースケール＋補正を"nearest"で実施
#     plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#     plt.axis("off")
#     plt.title("Training: " + str(labels[i]))
# plt.show()

# サポートベクターマシーン
clf = svm.SVC(gamma=0.001, C=100.0)
# サポートベクターマシーンによる訓練（6割のデータを使用、残りのデータは検証用）
clf.fit(digits.data[:n*6/10], digits.target[:n*6/10])

# train_data, test_data, train_target, test_target = train_test_split(digits.data, digits.traget, test_size=0.4, random_state=1)
# clf.fit(train_data, train_target)

# 正解
expected = digits.target[-n*4/10:]
# 予測
predicted = clf.predict(digits.data[-n*4/10:])
# 正解率
print metrics.classification_report(expected, predicted)
# 認識後のマトリックス
print metrics.confusion_matrix(expected, predicted)

# 予測と画像の対応（一部）
images = digits.images[-n*4/10:]
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.axis("off")
    plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess: " + str(predicted[i]))
plt.show()