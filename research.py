import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

df = pd.read_csv("/Users/siddhantagarwal/Desktop/Stats_copy_copy/2015-16_1-Table 1.csv")
exam = pd.read_csv("/Users/siddhantagarwal/Desktop/Stats_copy_copy/Exam results 2015-16.csv")
tot_pred = pd.read_csv("/Users/siddhantagarwal/Desktop/NAN.csv")

################################### MODEL FOR TOT_APP ##################################
no_nan = tot_pred.drop([10,27,30])
print(no_nan.head())

'''
X = no_nan[['area_sqkm', 'tot_population',
       'urban_population', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'tot_teachers']]
'''
X = no_nan[['tot_population','sc_population','area_sqkm','tot_teachers']]
y = no_nan['tot_app']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)

def evaluate(y_test,y_pred):
    print("MSE:",mean_squared_error(y_test,y_pred))
    print("R2:",r2_score(y_test,y_pred))
    print("MAE:",mean_absolute_error(y_test,y_pred))
    
evaluate(y_test,y_pred)
print()
'''
X = no_nan[['area_sqkm', 'tot_population',
       'urban_population', 'sexratio', 'sc_population',
       'st_population', 'literacy_rate', 'tot_teachers']]
'''
X = no_nan[['area_sqkm', 'tot_population','sc_population','tot_teachers']]
y = no_nan['pass_tot']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)

lm2 = LinearRegression()
lm2.fit(X_train,y_train)
y_pred = lm2.predict(X_test)

evaluate(y_test,y_pred)

X = tot_pred.iloc[[10,27,30]][['area_sqkm','st_population', 'tot_population','tot_teachers']]
pred = lm.predict(X)

X = tot_pred.iloc[[10,27,30]][['area_sqkm', 'tot_population','sc_population','tot_teachers']]
pred2 = lm2.predict(X)
for i in range(3):
    pred[i] = int(abs(pred[i]))
    pred2[i] = int(abs(pred2[i]))

print(pred)
print(pred2)

index = [10,27,30]
count = 0
for i in index:
    tot_pred.set_value(i,'tot_app',pred[count])
    tot_pred.set_value(i,'pass_tot',pred2[count])
    count += 1
print(tot_pred['tot_app'])
print(tot_pred['pass_tot'])

sns.distplot(no_nan['pass_percent'],bins = 33)
############## ASSIGNING POPULATION AND RATIOS FOR EACH STATE (STILL NEEDS WORK) #################
'''
tot_app = []
tot_pass = []

tot_pop = []


states = ['']
for i in df['STATNAME']:
    if not(i in states):
        states.append(i)
states.remove('')

state_pop = {}


for i in states:
    for j in df['STATNAME']:
        if j==i:
            x = df[df['STATNAME']==i]['TOTPOPULAT'].sum()
            state_pop[i] = [x]
            
print(state_pop)

values = state_pop.values()
values_list = list(values)

for i in range(len(states)):
    tot_pop.append(values_list[i][0])

print(tot_pop)
print()


count = 0
for i in states:
    x = (exam[exam['statname']==i]['tot_app'])/values_list[count][0]
    state_pop[i].append(x.values[0])
    x = (exam[exam['statname']==i]['pass_tot'])/values_list[count][0]
    state_pop[i].append(x.values[0])

print(state_pop)
'''

#################### USING RATIOS T0 FIND TOTAL APPEARED AND PASSED IN EACH DISTRCICT ####################

###############################################################

'''
print(type(pass_ratio))

states = ['']
for i in df['STATNAME']:
    if not(i in states):
        states.append(i)
states.remove('')

state_ratio = {}
count = 0
for i in states:
    state_ratio[i] = [app_ratio[count],pass_ratio[count]]
    count+=1
    
print(df.head())


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