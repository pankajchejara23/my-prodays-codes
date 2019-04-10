import pandas as pd
import prince

from sklearn import preprocessing


#data = pd.read_csv('datawithcat2.csv')

data_8nov = pd.read_csv('ProcessedData_ProjectDay_20171108.csv')
data_18oct = pd.read_csv('ProcessedData_ProjectDay_20171018.csv')
data_22nov = pd.read_csv('ProcessedData_ProjectDay_20171122.csv')

groups ={'physical':['disengaged','looking','talking','intTech','intRes','intExt'],'logs':['Accessed','Create','Open','Update']}


"""
Scenario -3
MFA train on 18 oct and 8 nov data
"""

# Deleting first three columns

data_18n8 = pd.concat([data_18oct,data_8nov]).reset_index(drop=True)

data_22nov.fillna(0,inplace=True)
data_18n8.to_csv('3_scenario_data_18n8.csv')


data_18n8.drop(data_18n8.columns[[0,1,2]],axis=1,inplace=True)
data_22nov.drop(data_22nov.columns[[0,1,2]],axis=1,inplace=True)

mfa1 = prince.MFA(groups=groups,n_components = 2)
mfa1 = mfa1.fit(data_18n8)

mfa_both = mfa1.transform(data_18n8)
mfa_22 = mfa1.transform(data_22nov)

mfa_both.to_csv('3_scenario_18n8.csv')
mfa_22.to_csv('3_scenario_22nov.csv')
