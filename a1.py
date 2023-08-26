import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
df=pd.read_excel("Lab Session1 Data.xlsx", sheet_name="Purchase data")
df=df.iloc[0:10,0:5]
aug_mat=df.iloc[0:10,1:5]
dim = np.linalg.matrix_rank(np.array(aug_mat))
print("Dimentionality : ",dim)
temp1=df.iloc[0:10,1:4]
A=np.array(temp1)
temp2=df.iloc[0:10,4:5]
C=np.array(temp2)
A_inverse=np.linalg.pinv(A)
X=np.matmul(A_inverse,C)
rank=np.linalg.matrix_rank(A)
print("Rank A : ",rank)