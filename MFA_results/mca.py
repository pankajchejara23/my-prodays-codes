import pandas as pd
import prince

from sklearn import preprocessing


#data = pd.read_csv('datawithcat2.csv')

data_8nov = pd.read_csv('ProcessedData_ProjectDay_20171108.csv')
data_18oct = pd.read_csv('ProcessedData_ProjectDay_20171018.csv')
data_22nov = pd.read_csv('ProcessedData_ProjectDay_20171122.csv')

groups ={'physical':['disengaged','looking','talking','intTech','intRes','intExt'],'logs':['Accessed','Create','Open','Update']}


"""
Scenario -1
Independent MFA on each data
"""

# Deleting first three columns
data_8nov.drop(data_8nov.columns[[0,1,2]],axis=1,inplace=True)
data_22nov.drop(data_22nov.columns[[0,1,2]],axis=1,inplace=True)
data_18oct.drop(data_18oct.columns[[0,1,2]],axis=1,inplace=True)


data_22nov_fillna = data_22nov.fillna(0)
data_22nov_dropna = data_22nov.dropna(0)



mfa1 = prince.MFA(groups=groups,n_components = 2)
mfa2 = prince.MFA(groups=groups,n_components = 2)
mfa3 = prince.MFA(groups=groups,n_components = 2)
mfa4 = prince.MFA(groups=groups,n_components = 2)

mfa1 = mfa1.fit_transform(data_8nov)
mfa2 = mfa2.fit_transform(data_22nov_dropna)
mfa3 = mfa3.fit_transform(data_18oct)
mfa4 = mfa4.fit_transform(data_22nov_fillna)

mfa1.to_csv('1_scenario_8nov.csv')
mfa2.to_csv('1_scenario_22nov_dropna.csv')
mfa3.to_csv('1_scenario_18oct.csv')
mfa4.to_csv('1_scenario_22nov_fillna.csv')
