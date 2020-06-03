import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv('/Users/siddhantagarwal/Documents/SidPythonCourses/db1.csv')
features = pd.read_csv('/Users/siddhantagarwal/Desktop/feature_columns.csv')

df = df[['STATNAME', 'DISTNAME', 'TCHTOTG', 'TCHTOTP', 'SCLSTOT', 'STCHTOT', 'ROADTOT', 'SPLAYTOT', 'SBNDRTOT', 'SGTOILTOT', 'SBTOILTOT', 'SWATTOT', 'SELETOT', 'SCOMPTOT', 'SRAMTOT', 'MDMTOT', 'KITTOT', 'KITSTOT', 'CLSTOT','PPFTCH', 'PPMTCH', 'PMTCH', 'PFTCH', 'PGRMTCH', 'PGRFTCH', 'GRMTCH', 'GRFTCH', 'PGCMTCH', 'PGCFTCH', 'PCMTCH', 'PCFTCH', 'tot_app', 'pass_tot','TCHMTOT']]


####### MODEL FOR ALL ##############
X = df.drop(['tot_app','pass_tot','STATNAME','DISTNAME'], axis = 1)
y = df['tot_app']

def Linear(size):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = size,random_state = 0)
    
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    y_pred = lm.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    
    print("RMSE: ",rmse)
    print("r2: ",r2)
    
    return rmse,r2

Linear(0.05)
performance_all = {}
performance_all['ALL'] = list(Linear(0.05))
print(performance_all)
print()
print()
print()

##### MODEL FOR INFRA #############
df1 = df[['tot_app','CLSTOT','pass_tot','ROADTOT','SRAMTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT']]

#df1.drop(570,inplace = True)

X = df1.drop(['tot_app','pass_tot'],axis = 1)
y = df1['tot_app']

Linear(0.05)
performance_all['infra'] = list(Linear(0.05))

print()
################## MODEL FOR ALL MDM ####################

print('MODEL FOR MDM')

df2 = df[['MDMTOT','KITTOT','KITSTOT','tot_app','pass_tot']]
X = df2.drop(['pass_tot','tot_app'],axis=1)
y = df2['tot_app']

Linear(0.15)

print()



