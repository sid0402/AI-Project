import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv('/Users/siddhantagarwal/Documents/SidPythonCourses/db1.csv')
df_dup = pd.read_csv('/Users/siddhantagarwal/Documents/SidPythonCourses/db1.csv')

df = df[['STATNAME','TCHTOT', 'TCHFTOT','DISTNAME', 'TCHTOTG', 'TCHTOTP', 'SCLSTOT', 'STCHTOT', 'ROADTOT', 'SPLAYTOT', 'SBNDRTOT', 'SGTOILTOT', 'SBTOILTOT', 'SWATTOT', 'SELETOT', 'SCOMPTOT', 'SRAMTOT', 'MDMTOT', 'KITTOT', 'KITSTOT', 'CLSTOT','PPFTCH', 'PPMTCH', 'PMTCH', 'PFTCH', 'PGRMTCH', 'PGRFTCH', 'GRMTCH', 'GRFTCH', 'PGCMTCH', 'PGCFTCH', 'PCMTCH', 'PCFTCH', 'tot_app', 'pass_tot','TCHMTOT','SCHTOT']]
df.drop(622,inplace = True)
X = df.drop(['tot_app','STATNAME','DISTNAME','pass_tot'],axis = 1)
y = df['tot_app']

df1 = df[['tot_app','CLSTOT','pass_tot','ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT','SRAMTOT']]

'''
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

####### MODEL FOR ALL #################
    
performance_all = []
for i in range(3):
    print("degree = ",i)
    PolyRegression(i)
    performance_all.append(PolyRegression(i))
    
print(performance_all)
print()
print()

######### MODEL FOR INFRA #############
print('PERFORMANCE FOR INFRA')

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
######################### RECOMMENDATIONS FOR TOT_APP ########################
ORIGINAL = []
PREDICTED = []
PERCENT_CHANGE = []
######## COMPUTERS ###########
print("sjejwirheojfejreoihf ",df_dup['pass_tot']*100/df_dup['tot_app'])
print()
print("COMPUTERS")
print()
df_dup = df_dup.iloc[622]

df_dup_infra = df_dup[['CLSTOT','ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT','SRAMTOT']]

print("SCOMPT RATIO TO SCHTOT: ",df_dup['SCOMPTOT']*100/df_dup['SCHTOT'])
print()
X = df1.drop(['tot_app','pass_tot'],axis = 1)
y = df1['pass_tot']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)

lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred1 = lm.predict(df_dup_infra.values.reshape(1,-1))

print("PREDICTED TOT_APP: ",int(y_pred1[0]))
ORIGINAL.append(y_pred1[0])

print("ORIGINAL SCOMPTOT: ",df_dup_infra['SCOMPTOT'])
df_dup_infra['SCOMPTOT'] = int(df_dup_infra['SCOMPTOT']*1.10)
print("NEW SCOMPTOT: ",df_dup_infra['SCOMPTOT'])
print()
y_pred2 = lm.predict(df_dup_infra.values.reshape(1,-1))
print('PREDICTED TOT APP WITH NEW SCOMPTOT: ',int(y_pred2[0]))
PREDICTED.append(y_pred2[0])

print("NEW SCOMPT RATIO TO SCHTOT: ",df_dup_infra['SCOMPTOT']*100/df_dup['SCHTOT'])
print("TOTAL NEW STUDENTS: ",y_pred2-y_pred1)
pc = ((y_pred2-y_pred1)/y_pred1)*100
PERCENT_CHANGE.append(pc[0])
print("PERCENT CHANGE: ",pc)

############# ELECTRICITY ################
print()
print('ELECTRICITY')
print()

print("PERCENTAGE THAT HAVE ELECTRICITY: ",df_dup['SELETOT']*100/df_dup['SCHTOT'])
print('SCHOOLS THAT DONT HAVE ELECTRICITY: ',df_dup['SELETOT']-df_dup['SCHTOT'])
df_dup_infra = df_dup[['CLSTOT','ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT','SRAMTOT']]
print()
print("ORIGINAL SELETOT: ",df_dup['SELETOT'])
df_dup_infra['SELETOT'] = int(df_dup_infra['SELETOT']*1.10)
print("NEW SELETOT: ",df_dup_infra['SELETOT'])
y_pred_ele = lm.predict(df_dup_infra.values.reshape(1,-1))

print("PREDICTED TOT APP WITH NEW SELETOT: ",int(y_pred_ele))
print()
print("NEW PERCENTAGE THAT HAVE ELECTRICITY: ",df_dup_infra['SELETOT']*100/df_dup['SCHTOT'])
pc = ((y_pred_ele-y_pred1)/y_pred1)*100
print('Percent Change: ',pc)

print('TOTAL NEW STUDENTS: ',y_pred_ele-y_pred1)

PERCENT_CHANGE.append(pc[0])
ORIGINAL.append(y_pred1[0])
PREDICTED.append(y_pred_ele[0])

print(PERCENT_CHANGE)
print(ORIGINAL)
print(PREDICTED)
print()

##########  TEACHERS ###########

print()
print('TEACHERS')
print()

#print('FFFJIFUFKDL: ',100*df_dup['STCHTOT']/df_dup['SCHTOT'])
#print("jdnnfrk: ",100*df_dup['TCHTOTG']/df_dup['TCHTOT'])

df_dup_tch = df_dup[['TCHTOT','PGRFTCH','PGRMTCH','GRFTCH']]
df_tch = df[['TCHTOT','PGRFTCH','PGRMTCH','GRFTCH','tot_app','pass_tot']]

#print("YOOOOO: ",100*df_dup['TCHFTOT']/df_dup['TCHTOT'])

X = df_tch.drop(['tot_app','pass_tot'],axis=1)
y = df_tch['pass_tot']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)

lm = LinearRegression()
lm.fit(X_train,y_train)

y_pred_org = lm.predict(df_dup_tch.values.reshape(1,-1))
print("PREDICTED TOTAL APPEARANCE: ",y_pred_org)
ORIGINAL.append(y_pred_org[0])

print("ORIGINAL TCHTOT: ",df_dup_tch['TCHTOT'])
df_dup_tch['TCHTOT'] = int(df_dup_tch['TCHTOT']*1.10)
print('NEW TCHTOT: ',df_dup_tch['TCHTOT'])

y_pred_tch = lm.predict(df_dup_tch.values.reshape(1,-1))
PREDICTED.append(list(y_pred_tch)[0])

print("PREDICTED TOT_APP WITH NEW TCHTOT: ",y_pred_tch)
#print('NEW TCHTOTG %: ',100*df_dup_tch['TCHTOT']/df_dup['TCHTOT'])
pc = (100*(y_pred_tch-y_pred_org))/y_pred_org
PERCENT_CHANGE.append(list(pc)[0])
print("PERCENT CHANGE: ",pc)
print("TOTAL NEW STUDENTS: ",y_pred_tch-y_pred_org)
print()
print(PERCENT_CHANGE)
print(ORIGINAL)
print(PREDICTED)
print()

print(df_dup['TCHFTOT']/df_dup['TCHTOT'])

#### VISUALISATION ##########
'''
labels = ['COMPUTERS','ELECTRICITY','GOVERNMENT TEACHERS']

fig,(ax1,ax2) = plt.subplots(2,figsize=(11,11))

x = np.arange(len(labels))
width = 0.35

ax1.bar(x - width/2, ORIGINAL, width, label='ORIGINAL',color = 'cornflowerblue')
ax1.bar(x + width/2, PREDICTED, width, label='PREDICTED',color = 'mediumblue')
ax1.legend()
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel('RECOMMENDATIONS')
ax1.set_ylabel('TOTAL APPEARANCE IN EXAMS')
ax1.set_title('ORIGINAL VS PREDICTED TOT_APP')

ax2.bar(x,PERCENT_CHANGE,color = 'red',width = width)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel('RECOMMENDATIONS')
ax2.set_ylabel('PERCENTAGE')
ax2.set_title('PERCENT CHANGE FOR EACH RECOMMENDATION')

##################################

predicted = []
change = []

fig, ax = plt.subplots(3)

def graph(df,name,org):
    
    lm = LinearRegression()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)
    lm.fit(X_train,y_train)
    
    for i in range(21):
        df[name] = int(df[name]*(1+i/100))
        change.append(i)
        y_pred = lm.predict(df.values.reshape(1,-1))
        predicted.append(list((y_pred-4623551)/4623551)[0])
        
    return change,predicted

df_dup = df_dup.iloc[622]
df_dup_infra = df_dup[['CLSTOT','ROADTOT','SPLAYTOT','SBNDRTOT','SGTOILTOT','SBTOILTOT','SWATTOT','SELETOT','SCOMPTOT','SRAMTOT']]
X = df1.drop(['tot_app','pass_tot'],axis = 1)
y = df1['tot_app']

changes,predicted = graph(df_dup_infra,'SCOMPTOT',4623551)
#ax.plot(changes,predicted,color='red')

print(change)
print(predicted)

print("YOOOOO: ",100*df_dup['KITS3TOT']/df_dup['MDMTOT'])


models = list(zip(*[['Infrastructure', 'Infrastructure', 'Teachers', 'Teachers',
                     'Mid-Day Meal', 'Mid-Day Meal'],
                      ['tot_app', 'pass_tot', 'tot_app', 'pass_tot',
                       'tot_app', 'pass_tot']]))
columns = list(zip(*[['Linear Regression','Linear Regression','Polynomial Regression, Polynomial Regression'],['R2','RMSE','R2','RMSE']]))
    
index = pd.MultiIndex.from_tuples(models, names=['Category', 'Target Variable'])
columns = pd.MultiIndex.from_tuples(columns)

values = np.array([[-0.75545684, -0.317909  ]
 [ 0.46671742,  0.5249783 ]
 [ 1.29919965, -2.04170598]
 [-0.02811123,  0.35946973]
 [-0.1015853  , 0.32986674]
 [-0.3927652 , -1.13493111]])

df = pd.DataFrame(np.random.randn(6,2), index=index, columns=['A','B'])
print(df)

plt.scatter(df_tch['GRFTCH'],df_tch['tot_app'])

X = df.drop(['STATNAME','DISTNAME','pass_tot'],axis = 1)
cor_dict = {}
for j in X.columns:
    x = X[j]
    y = X['tot_app']
    cor = np.corrcoef(x,y)[0,1]
    cor_dict[j] = cor

for key, value in sorted(cor_dict.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
'''