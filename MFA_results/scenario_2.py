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
data_all = pd.concat([data_18oct,data_8nov,data_22nov]).reset_index(drop=True)
data_all.fillna(0,inplace=True)
data_all.to_csv('2_scenario_data.csv')

data_all.drop(data_all.columns[[0,1,2]],axis=1,inplace=True)


mfa1 = prince.MFA(groups=groups,n_components = 2)
mfa1 = mfa1.fit_transform(data_all)


mfa1.to_csv('2_scenario_all.csv')
