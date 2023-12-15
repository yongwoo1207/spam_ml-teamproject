import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score



gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
cnb = ComplementNB()

file_path = "/home/jihyun/"
data = pd.read_csv('emails.csv')

data.head() #첫부분
data.tail() #끝부분
data.info() #데이터 프레임 정보
data.isnull().sum() #결측치 확인

label = data.pop('Prediction') #레이블 분리

features = data
#features.head() //features 확인
#label.head() //label 확인

email_number = features.pop('Email No.')
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=None, stratify=label)
#훈련용, 테스트용  나누기

#모델학습
gnb.fit(X_train,y_train)
mnb.fit(X_train,y_train)
bnb.fit(X_train,y_train)
cnb.fit(X_train, y_train)   

#테스트용 데이터에 대한 예측
predGau = gnb.predict(X_test)
predMul = mnb.predict(X_test)
predBer = bnb.predict(X_test)
predCom = cnb.predict(X_test)

print('\n')

#각 모델의 정확도 계산 및 출력
print('Gaussian :', accuracy_score(predGau,y_test))
print('Multinomial :',accuracy_score(predMul,y_test))
print('Bernoulli :',accuracy_score(predBer,y_test))
print('Complement :',accuracy_score(predCom,y_test))

# 각 모델의 정확도 계산|
accuracy_scores = {
    'Gaussian': accuracy_score(predGau, y_test),
    'Multinomial': accuracy_score(predMul, y_test),
    'Bernoulli': accuracy_score(predBer, y_test),
    'Complement': accuracy_score(predCom, y_test)
}

# 가장 높은 정확도를 가진 모델과 그 정확도 출력
best_model = max(accuracy_scores, key=accuracy_scores.get)
best_accuracy = accuracy_scores[best_model]

print(f"Highest Accuracy: {best_accuracy} from {best_model}")

# 혼동 행렬
conf_matrix = confusion_matrix(y_test, predMul)
print("Confusion Matrix:")
print(conf_matrix)

# 정밀도와 재현율
precision = precision_score(y_test, predMul)
recall = recall_score(y_test, predMul)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

