import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv("/Users/siddhantagarwal/Desktop/Stats_copy_copy/2015-16_1-Table 1.csv")
exam = pd.read_csv("/Users/siddhantagarwal/Desktop/Stats_copy_copy/Exam results 2015-16.csv")
tot_pred = pd.read_csv("/Users/siddhantagarwal/Desktop/NAN.csv")

################ VISUALISATION #####################
no_nan = tot_pred.drop([10,27,30])

X = no_nan[['area_sqkm', 'tot_population',
       'urban_population', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'tot_teachers']]

cor_dict = {}
values = []
labels1 = []
for j in X.columns:
    x = no_nan[j]
    y = no_nan['tot_app']
    cor = np.corrcoef(x,y)[0,1]
    values.append(cor)
    labels1.append(j)
    cor_dict[j] = cor

for key, value in sorted(cor_dict.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))

print(cor_dict)
abc = pd.DataFrame(cor_dict,index = [0,1])
'''
fig = plt.figure()
ax = fig.add_axes([2,1,1.5,2])

abc = abc.head(1)

labels = np.arange(len(values))
print(labels)

ax.bar(labels,values,color = 'red')
ax.set_xticklabels(labels1)
plt.xticks(rotation = 45,color = 'black')
ax.set_xlabel('FEATURES')
ax.set_ylabel('CORRELATION COEFFECIENT')
ax.set_title('CORRELATION OF EACH FEATURE WITH TOT_APP')
'''  
X = no_nan[['area_sqkm', 'tot_population',
       'urban_population', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'tot_teachers','tot_app','pass_tot']]
fig = plt.figure()
ax = fig.add_axes([2,1,1.5,2])
sns.heatmap(X.corr(),cmap="YlGnBu")
################################### MODEL FOR TOT_APP ##################################
no_nan = tot_pred.drop([10,27,30])
#print(no_nan.head())

'''
X = no_nan[['area_sqkm', 'tot_population',
       'urban_population', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'tot_teachers']]
'''
X = no_nan[['tot_population','sc_population','area_sqkm','tot_teachers']]
y = no_nan['tot_app']

def evaluate(y_test,y_pred):
    print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))
    print("R2:",r2_score(y_test,y_pred))
    print("MAE:",mean_absolute_error(y_test,y_pred))

test_sizes = [0.05,0.1,0.15,0.2,0.25]
for i in test_sizes:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = i,random_state = 0)
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    y_pred = lm.predict(X_test)
    print()
    print('test_size = ',i)
    print()
    evaluate(y_test,y_pred)


'''
X = no_nan[['area_sqkm', 'tot_population',
       'urban_population', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'tot_teachers']]
'''
X = no_nan[[ 'tot_population','tot_teachers']]
y = no_nan['pass_tot']

test_sizes = [0.05,0.1,0.15,0.2,0.25,0.3]
for i in test_sizes:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = i,random_state = 0)
    lm2 = LinearRegression()
    lm2.fit(X_train,y_train)
    y_pred = lm2.predict(X_test)
    print()
    print('test_size = ',i)
    print()
    evaluate(y_test,y_pred)

#evaluate(y_test,y_pred)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.05,random_state = 0)
lm2 = LinearRegression()
lm2.fit(X_train,y_train)
y_pred = lm2.predict(X_test)

X = tot_pred.iloc[[10,27,30]][['area_sqkm','st_population', 'tot_population','tot_teachers']]
pred = lm.predict(X)

X = tot_pred.iloc[[10,27,30]][['tot_population','tot_teachers']]
pred2 = lm2.predict(X)
for i in range(3):
    pred[i] = int(abs(pred[i]))
    #pred2[i] = int(abs(pred2[i]))
    pred2[i] = abs(pred2[i])

#print(pred)
print("SHSUJIOSHISA ",pred2)

index = [10,27,30]
count = 0
pred_values = []
for i in index:
    tot_pred.set_value(i,'tot_app',pred[count])
    #x = (pred2[count]*pred[count])
    tot_pred.set_value(i,'pass_tot',pred2[count])
    tot_pred.set_value(i,'pass_percent',pred2[count])
    pred_values.append(x)
    count += 1

abc = tot_pred.iloc[[10,27,30]]
#print(tot_pred['tot_app'])
#print(tot_pred['pass_tot'])
############ VISUALISATION ##################

fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
#ax1 = sns.scatterplot(x = 'tot_population', y = 'pass_tot', data = no_nan,marker = "D",palette = 'deep' )
#ax1 = sns.scatterplot(x = 'tot_population', y = 'pass_tot', data = abc, color = 'red',marker = 'D')
#ax1.scatter(no_nan['tot_population'],no_nan['pass_tot'],marker = '.')
#ax1.scatter(abc['tot_population'],abc['pass_tot'],color = 'r',marker='o')
#ax2.scatter(no_nan['area_sqkm'],no_nan['pass_tot'],marker = '.')
#ax2.scatter(abc['area_sqkm'],abc['pass_tot'],color = 'red',marker = 'o')
sns.set_style('darkgrid')
ax = sns.scatterplot(x = 'tot_population', y = 'pass_tot', data = abc, color = 'red',marker = 'D',s = 400)
ax = sns.scatterplot(x = 'tot_population', y = 'pass_tot', data = no_nan,marker = "o",s = 200,color = 'deepskyblue')
ax.legend()

fig = plt.figure()
ax2 = fig.add_axes([1,1,1,1])
ax2 = sns.scatterplot(x = 'area_sqkm', y = 'pass_tot', data = abc, color = 'red',marker = 'D',s = 400)
ax2 = sns.scatterplot(x = 'area_sqkm', y = 'pass_tot', data = no_nan,marker = "o",s = 200,color = 'deepskyblue')

############# ASSIGNING POPULATION AND RATIOS FOR EACH STATE (STILL NEEDS WORK) #################
'''
tot_app = []
tot_pass = []

tot_pop = []


states = ['']
for i in df['STATNAME']:
    if not(i in states):
        states.append(i)
states.remove('')
states = states[:35]

print(states)
print()
print(len(states))


state_pop = {}
populat = []
for i in states:
    for j in states:
        if j==i:
            x = df[df['STATNAME']==i]['TOTPOPULAT'].sum()
            state_pop[i] = [x]
            populat.append(x)
            
#print(state_pop)

values = state_pop.values()
values_list = list(values)
#print(values_list)

for i in range(len(states)):
    tot_pop.append(values_list[i][0])

#print(tot_pop)
print()

for i in range(len(tot_pop)):
    tot_pred['tot_population'][i] = tot_pop[i]
    tot_pred['statname'][i] = states[i]


#x = (tot_pred[tot_pred['statname'] == 'TAMIL NADU']['tot_app'])
#rint(x)


#print(tot_pred[['tot_population','statname']])
#CREATING DICTIONARY CONTAINING POPULATION, APP RATIO AND PASS RATIO FOR EACH STATE
count = 0
for i in states:
    x = (tot_pred[tot_pred['statname']==i]['tot_app'])/tot_pop[count]
    state_pop[i].append(x.values[0]) #appending the tot_app to tot_pop ratio
    x = (exam[exam['statname']==i]['pass_tot'])/tot_pop[count]
    state_pop[i].append(x.values[0]) #appending tot pass to tot_pop ratio
    count+=1

#DIAGNOSTIC TO CHECK STATE POP
for i in states:
    x = state_pop[i][1]
    y = state_pop[i][2]
    if not(x>0) or not(y>0):
        print(i)

x = tot_pred[tot_pred['statname']=='SIKKIM']['pass_tot']
y = tot_pred[tot_pred['statname']=='SIKKIM']['tot_population']
z = x/(y)
state_pop['SIKKIM'][2] = z.values[0]

x = tot_pred[tot_pred['statname']=='ANDHRA PRADESH']['pass_tot']
y = tot_pred[tot_pred['statname']=='ANDHRA PRADESH']['tot_population']
z = x/y
state_pop['ANDHRA PRADESH'][2] = z.values[0]
#print(state_pop['ANDHRA PRADESH'])


#################### USING RATIOS T0 FIND TOTAL APPEARED AND PASSED IN EACH DISTRCICT ####################

x = df[df['STATNAME']=='ANDHRA PRADESH']['DISTNAME']
print(x)
x = df[(df["STATNAME"]=='ANDHRA PRADESH')&(df['DISTNAME']=='SRIKAKULAM')]['TOTPOPULAT']
print(x)

x = len(list(df.index))

TOT_APP = []
PASS_TOT = []

for i in states:
    x = df[df['STATNAME']==i]['DISTNAME']
    for j in x:
        y = df[(df["STATNAME"]==i)&(df['DISTNAME']==j)]['TOTPOPULAT']
        TOT_APP.append(int(y*state_pop[i][1]))
        #l = df[(df["STATNAME"]==i)&(df['DISTNAME']==j)]['tot_app'] 
        #l = y * state_pop[i][1]
        PASS_TOT.append(int(y*state_pop[i][2]))
        #m = df[(df["STATNAME"]==i)&(df['DISTNAME']==j)]['pass_tot'] 
        #m = y * state_pop[i][2]

#print(TOT_APP)
print()

df['tot_app'] = TOT_APP
df['pass_tot'] = PASS_TOT

#print(df[['DISTNAME','tot_app','pass_tot']])

for i in range(len(TOT_APP)):
    if pd.isna(TOT_APP[i]) or pd.isna(PASS_TOT[i]) is True:
        print(df['DISTNAME'][i])
    
#print(df.columns)
df.to_csv('db1.csv')


print("STATE POP: ", state_pop)
print()
print("POPULAT: " ,sorted(populat))

for i in sorted(populat):
    

for i in states:
    for j in df['DISTNAME']:
        x = (df[(df['STATNAME']==i)&(df['DISTNAME']==j)]['TOTPOPULAT'])*state_ratio[i][0]
        tot_app.append(list(x))
        y = (df[(df['STATNAME']==i)&(df['DISTNAME']==j)]['TOTPOPULAT'])*state_ratio[i][1]
        tot_pass.append(list(y))
        
tot_app = [x for x in tot_app if x != []]
tot_pass = [x for x in tot_pass if x != []]
count = 0
for i in tot_app:
    tot_app[count] = int(i[0])
    count += 1
    
count = 0
for i in tot_pass:
    tot_pass[count] = int(i[0])
    count += 1
    
print(tot_app)
print(len(tot_app))
print(len(tot_pass))
        

for i in exam['statname']:
    for j in df['STATNAME']:
        if j == i:
            for k in df['DISTNAME']:
                x = (df[(df['STATNAME']==i)&(df['DISTNAME']==k)]['TOTPOPULAT']) * (exam[exam['statname']==i]['appeared_ratio'])
                tot_app.append(x)

####################### EXTRA #################################################

print(df.columns)
print(df.head())
df1 = df[['SCLSTOT',	'STCHTOT',	'SPLAYTOT'	,'SGTOILTOT'	,'SBTOILTOT'	,'SWATTOT','KITTOT','MDMTOT']]
print(df1.head())

mdm_corr = pd.DataFrame(df1.corr())['MDMTOT']
print(mdm_corr)

x = ['SCLSTOT',	'STCHTOT',	'SPLAYTOT'	,'SGTOILTOT'	,'SBTOILTOT'	,'SWATTOT','KITTOT']
y = []
for i in x:
    y.append(mdm_corr[i])

x = ['Single Classroom Schools','Single Teacher Schools','Schools with playground facility','Schools with Girls Toilet','Schools with Boys Toilet','Schools with Drinking Water','Schools with Kitchen Shed']
fig = plt.figure()
ax = fig.add_axes([1,1,1.5,1.5])
ax.set_title('Features and their correlations with Schools availing MDM')
ax.set_ylabel('Features')
ax.set_xlabel('Correlation coefficient')
ax.barh(x,y,color = 'red')

plt.show()

'''
