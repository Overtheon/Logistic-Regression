import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("C:\\Users\\ASUS\\Desktop\\逻辑回归数据集.csv")
#训练
# 假设X是特征数据，y是标签数据,test_size=0.2表示有 20%的数据用于测试，random_state控制数据划分的随机性,保证每次分割结果相同(相当于随机种子random.seed())
X_train, X_test, y_train, y_test = train_test_split(df[["Age","EstimatedSalary"]].values, df.Purchased, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
#评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率：{accuracy}")
#预测
new_data=[[20,30000],[50,50000]]
pred=model.predict_proba(new_data)
print(pred)
#sklearn库版本，一般版本越高模型准确率越高
#print(sklearn.__version__)