# coding: UTF-8
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

"""
入力値：
    4日、3日、2日、前日の株値
正解値：
    当日の株が上昇 -> 1
    当日の株が下降 -> 0
"""

# ファイルの読み込み
stock_data = []
stock_data_file = open("stock_price","r")
for line in stock_data_file:
    line = line.rstrip()
    stock_data.append(float(line))
stock_data_file.close()



# 株価の上昇率を算出、おおよそ-1.0 ~ 1.0の範囲に収まるように設定
modified_data = []
count_s = len(stock_data)
for i in range(1, count_s):
    modified_data.append(float(stock_data[i] - stock_data[i-1])/ float(stock_data[i-1] * 20))
count_m = len(modified_data)

# 前日までの4日間の上昇率のデータ
successive_data = []
# 正解値 株価上昇：1 株価低下：0
answers = []
for i in range(4, count_m):
    successive_data.append([modified_data[i-4], modified_data[i-3], modified_data[i-2], modified_data[i-1]])
    if modified_data[i] > 0:
        answers.append(1)
    else:
        answers.append(0)

n = len(successive_data)

clf = svm.LinearSVC()
clf.fit(successive_data[:n*75/100],answers[:n*75/100])

# テスト用データ
# 正解
expected = answers[-n*25/100:]
# 予測
predicted = clf.predict(successive_data[-n*25/100:])

# print expected[-10:]
# print predicted[-10:]

# 正解率の計算
correct = 0.0
wrong = 0.0
for i in range(n*25/100):
    if expected[i] == predicted[i]:
        correct += 1
    else:
        wrong += 1

print "正解率: " + str(correct / (correct + wrong) * 100) + "%"