import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_excel("Lab Session1 Data.xlsx", sheet_name="IRCTC Stock Price")
col=df.iloc[:,3:4]
mean_p=np.mean(np.array(col))
var_p=np.var(np.array(col))
print('Mean :', mean_p)
print("Variance : ",var_p)
A=df.loc[df['Day']=='Wed']
A1=A.iloc[:,3:4]
A2=np.mean(np.array(A1))
B=df.loc[df['Month']=='Apr']
B1=B.iloc[:,3:4]
mean_apr=np.mean(np.array(B1))
print('Mean on wednesday :',A2 )
print('Mean in april:', mean_apr)
dta=df.iloc[:,8:9]
dta1=np.array(dta)
dta2=np.array(A.iloc[:,8:9])
n1=len(dta1)
n2=0
for i in range(0,n1):
    if dta1[i]<0:
        n2=n2+1
loss=n2/n1
print("Probability of loss : ",loss)
cout=0
n1=len(dta2)
for i in range(0,n1):
    if dta2[i]<0:
        cout=cout+1
prof=cout/n1
print("Probability of profit on wednesday : ",prof)
weddata = df[df['Day'] == 'Wed']
wedprofit = np.mean(weddata['Chg%'] > 0)
wedprob = np.mean(df['Day'] == 'Wed')
cdprob = (wedprofit / wedprob)
print('Conditional Probability on Wednesday is:', cdprob)
plt.scatter(df['Day'], df['Chg%'])
plt.xlabel('Day')
plt.ylabel('Chg%')
plt.title('Chg% vs Day')
plt.show()