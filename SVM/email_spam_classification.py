import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("/kaggle/input/email-spam-classification-dataset-csv/emails.csv")

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# 모델 학습
model_SVM = SVC(kernel='linear')
model_SVM_nonlinear = SVC(kernel='poly', degree=3)

model_SVM.fit(X_train, y_train)
model_SVM_nonlinear.fit(X_train, y_train)

# 테스트용 데이터 예측
y_pred_SVM = model_SVM.predict(X_test)
y_pred_SVM_nonlinear = model_SVM_nonlinear.predict(X_test)

# 모델 평가
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
precision_SVM = precision_score(y_test, y_pred_SVM)
recall_SVM = recall_score(y_test, y_pred_SVM)

accuracy_SVM_nonlinear = accuracy_score(y_test, y_pred_SVM_nonlinear)
precision_SVM_nonlinear = precision_score(y_test, y_pred_SVM_nonlinear)
recall_SVM_nonlinear = recall_score(y_test, y_pred_SVM_nonlinear)


# 평가 출력
results["SVM"] = {"accuracy": accuracy_SVM, 
                  "precision": precision_SVM, 
                  "recall": recall_SVM, 
                 }
results["SVM_nonlinear"] = {"accuracy": accuracy_SVM_nonlinear, 
                  "precision": precision_SVM_nonlinear, 
                  "recall": recall_SVM_nonlinear, 
                 }
results
