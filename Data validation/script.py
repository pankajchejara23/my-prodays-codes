import pandas as pd
import prince


data = pd.read_csv('data_two_days.csv')
data2 = pd.read_csv('ProcessedData_ProjectDay_20171108.csv')

data.drop(data.columns[0],axis=1,inplace=True)
data2.drop(data2.columns[0],axis=1,inplace=True)

raw_data = data.drop(['timestamp','group'],axis=1)

raw_data2 = data2.drop(['timestamp','group'],axis=1)

groups ={'physical':['disengaged','looking','talking','intTech','intRes','intExt'],'logs':['Accessed','Create','Open','Update']}



mfa = prince.MFA(groups=groups,n_components = 2)
mfa = mfa.fit(raw_data)

print('columns in 22 data',data2.columns)
print('columns in first two day data',data.columns)

mfadf = mfa.row_coordinates(raw_data)
mfadf.to_csv('mfa_two_day.csv')

mfadf1 = mfa.row_coordinates(raw_data2)
mfadf1.to_csv('mfa_third_day.csv')
