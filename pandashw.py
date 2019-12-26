import pandas as pd 

#reading the csv file  
df = pd.read_csv("https://www.ldeo.columbia.edu/~rpa/usgs_earthquakes_2014.csv")
print(df.head())

#Removing certain, irrelevant columns
df = df.drop(['time','latitude', 'longitude', 'magType', 'nst','net','id','updated','gap', 'dmin', 'rms','depth'],axis = 1)


#finding the 10 strongest earthquakes
print("the ten strongest earthquakes are:")
df = df[df['type']=='earthquake']
df = df.sort_values('mag',ascending = False)
print(df.head(10))

#finding the 10 weakest earthquakes
print()
print("the ten weakest earthquakes are:")
df = df[df['type']=='earthquake']
df = df.sort_values('mag',ascending = True)
print(df.head(10))