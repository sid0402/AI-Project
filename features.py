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
decoy = pd.read_csv('/Users/siddhantagarwal/Documents/SidPythonCourses/db1.csv')


print(df.describe())
print()
decoy = decoy[['ENRTOT','STATNAME', 'DISTNAME', 'TCHTOTG', 'TCHTOTP', 'SCLSTOT', 'STCHTOT', 'ROADTOT', 'SPLAYTOT', 'SBNDRTOT', 'SGTOILTOT', 'SBTOILTOT', 'SWATTOT', 'SELETOT', 'SCOMPTOT', 'SRAMTOT', 'MDMTOT', 'KITTOT', 'KITSTOT', 'CLSTOT','PPFTCH', 'PPMTCH', 'PMTCH', 'PFTCH', 'PGRMTCH', 'PGRFTCH', 'GRMTCH', 'GRFTCH', 'PGCMTCH', 'PGCFTCH', 'PCMTCH', 'PCFTCH', 'tot_app', 'pass_tot','TCHMTOT','pass_percent']]

print(df.head())

df = df[['STATNAME','TCHTOT', 'TCHFTOT','DISTNAME', 'TCHTOTG', 'TCHTOTP', 'SCLSTOT', 'STCHTOT', 'ROADTOT', 'SPLAYTOT', 'SBNDRTOT', 'SGTOILTOT', 'SBTOILTOT', 'SWATTOT', 'SELETOT', 'SCOMPTOT', 'SRAMTOT', 'MDMTOT', 'KITTOT', 'KITSTOT', 'CLSTOT','PPFTCH', 'PPMTCH', 'PMTCH', 'PFTCH', 'PGRMTCH', 'PGRFTCH', 'GRMTCH', 'GRFTCH', 'PGCMTCH', 'PGCFTCH', 'PCMTCH', 'PCFTCH', 'tot_app', 'pass_tot','TCHMTOT']]
print(df.columns)

X = df.drop(['tot_app','STATNAME','DISTNAME','pass_tot'],axis = 1)
y = df['tot_app']

##### MODEL WITH ALL FEATURES#####################
def PolyRegression(degree):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)
    
    
    poly = PolynomialFeatures(degree = degree)
    X_train_poly = poly.fit_transform(X_train)
    
    lm = LinearRegression()
    lm.fit(X_train_poly,y_train)
    y_pred = lm.predict(poly.fit_transform(X_test))

    
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    print("rmse:",rmse)
    print("R2: ",r2)
    
    return rmse,r2
  
performance_all = []
for i in range(5):
    print("degree = ",i)
    PolyRegression(i)
    performance_all.append(PolyRegression(i))

print(performance_all)
print()
print()
print()

#### MODEL FOR INFRASTRUCTURE (RMSA) ####################

#df1 = df[['tot_app','pass_tot','ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT']]
df1 = df[['tot_app','CLSTOT','pass_tot','ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT','SRAMTOT']]

X = df1.drop(['tot_app','pass_tot'],axis = 1)
y = df1['pass_tot']

print("PERFORMANCE FOR INFRA")
performance_infra = []
for i in range(5):
    print("degree = ",i)
    PolyRegression(i)
    performance_infra.append(PolyRegression(i))

print(performance_infra)
print()
print()
############ MODEL FOR MDM ##################
df_mdm = df[['MDMTOT', 'KITTOT', 'KITSTOT']]
X = df_mdm
print("PERFORMANCE FOR MDM")
performance_mdm = []
for i in range(5):
    print("degree = ",i)
    PolyRegression(i)
    performance_mdm.append(PolyRegression(i))

print(performance_mdm)
print()
print()
############ MODEL FOR TEACHERS ###################
print('PERFORMANCE FOR TEACHERS')
df_tch = df[['TCHTOTG','PGRFTCH','PGRMTCH','GRFTCH']]
X = df_tch
performance_tch = []
for i in range(5):
    print("degree = ",i)
    PolyRegression(i)
    performance_tch.append(PolyRegression(i))

print(performance_tch)
print()
print()

'''
decoy = decoy.drop(['STATNAME', 'DISTNAME'], axis = 1)

cor_dict = {}
for j in decoy.columns:
    x = decoy[j]
    y = decoy['pass_tot']
    cor = np.corrcoef(x,y)[0,1]
    cor_dict[j] = cor

for key, value in sorted(cor_dict.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
'''

'''
performances = {}
labels = ['ALL','INRASTRUCTURE','MDM','TEACHERS']
vals = [performance_all, performance_infra,performance_mdm,performance_tch]

for i in range(4):
    performances[labels[i]] = vals[i]

print(pd.DataFrame(performances).head())

a = ['ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT']

for i in a:
    plt.scatter(df[i],df['tot_app'])
    plt.xlabel(i)
    plt.ylabel("total appeared")
    plt.show()
    sns.lmplot(i,'tot_app',data = df1)


X = df['MDMTOT'].values.reshape(-1,1)
y = df['tot_app']
'''



'''
def PolyRegression2(degree,size):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = size,random_state = 0)
    
    
    poly = PolynomialFeatures(degree = degree)
    X_train_poly = poly.fit_transform(X_train)
    
    lm = LinearRegression()
    lm.fit(X_train_poly,y_train)
    y_pred = lm.predict(poly.fit_transform(X_test))

    
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    print("rmse:",rmse)
    print("R2: ",r2)
    
    y_pred = lm.predict(poly.fit_transform(X))
    
    plt.scatter(X,y)
    plt.plot(X,y_pred,color = 'red')
    
    return rmse,r2


PolyRegression2(4,0.05)
'''
'''
correlations = {}

y = df['TCHTOTG']

df2 = df.drop(['STATNAME', 'DISTNAME'], axis = 1)

for j in df2.columns:
    cor_dict = {}
    for i in df2.columns: 
        x = df2[i]
        y = df2[j]
        cor = np.corrcoef(x,y)[0,1]
        cor_dict[i] = cor
    correlations[str(j)] = cor_dict
    
#print(correlations)

corr_df = pd.DataFrame(correlations)

print(corr_df)

x = corr_df['tot_app']

fig = plt.figure(1)
ax = fig.add_axes([2,1,1.5,2])

rows = list(corr_df.index.values)
values = list(corr_df['tot_app'].values)
values2 = list(corr_df['pass_tot'].values)

ax.bar(rows,values,color = 'cornflowerblue')

#ax.set_facecolor('turquoise')
plt.xticks(rotation = 45,color = 'black')

decoy = decoy.drop(['STATNAME', 'DISTNAME'], axis = 1)

cor_dict = {}
for j in decoy.columns:
    x = decoy[j]
    y = decoy['tot_app']
    cor = np.corrcoef(x,y)[0,1]
    cor_dict[j] = cor

for key, value in sorted(cor_dict.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))

decoy = decoy.drop(['STATNAME', 'DISTNAME'], axis = 1)

cor_dict = {}
for j in decoy.columns:
    x = decoy[j]
    y = decoy['tot_app']
    cor = np.corrcoef(x,y)[0,1]
    cor_dict[j] = cor

for key, value in sorted(cor_dict.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
'''