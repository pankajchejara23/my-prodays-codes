import pandas as pd
import prince

from sklearn import preprocessing


#data = pd.read_csv('datawithcat2.csv')

data = pd.read_csv('ProcessedData_ProjectDay_20171122.csv')
#data1 = pd.read_csv('Data_16_original.csv')
#data = pd.read_csv('8th16th.csv')

# Deleting the first columns of sequence number
data.drop(data.columns[0],axis=1,inplace=True)

# Printing basic info of dataset
#print ('8th Nov dataset:',data.shape)
#print ('16th Nov:',data1.shape)
#print('both days:',data.shape)

# Printing Column names
#print([a for a in data.columns])

#print(data['timestamp'])

raw_data = data.drop(['timestamp','group'],axis=1)
raw_data = data.fillna(value=0,method=None,axis=1)
#raw_data1 = data1.drop(['timestamp','group'],axis=1)

# Stadardizing the dataset
#std_rawdata = preprocessing.StandardScaler().fit_transform(raw_data)
#std_rawdata1 = preprocessing.StandardScaler().fit_transform(raw_data1)

#mca = prince.MCA(n_components =2, n_iter=3,copy=True,engine='auto')
#mca = mca.fit(raw_data)

#mca1 = prince.MCA(n_components =2, n_iter=3,copy=True,engine='auto')
#mca1 = mca1.fit(raw_data1)

groups ={'physical':['disengaged','looking','talking','intTech','intRes','intExt'],'logs':['Accessed','Create','Open','Update']}



mfa = prince.MFA(groups=groups,n_components = 2)
mfa = mfa.fit(raw_data)

#mfa1 = prince.MFA(groups=groups,n_components = 2)
#mfa1 = mfa.fit(raw_data1)

#mcadf = mca.row_coordinates(raw_data)
#mcadf.to_csv('mcaresult_both.csv')

#mcadf1 = mca1.row_coordinates(raw_data1)
#mcadf1.to_csv('mcaresult_16th.csv')



mfadf = mfa.row_coordinates(raw_data)
mfadf.to_csv('mfa_ProjectDay_20171122_fillna.csv')

#mfadf1 = mfa1.row_coordinates(raw_data1)
#mfadf1.to_csv('mfaresult_16th.csv')
