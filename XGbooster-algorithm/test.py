import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(100).reshape((10, 10)),range(10)
print(X)
print(y)
print("end")

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2,random_state=123)

print(X_train,X_test)