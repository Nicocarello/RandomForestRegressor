import pandas as pd

data = pd.read_csv('D:/Anaconda/datasets/boston/Boston.csv')

colnames = data.columns.values.tolist()

predictors = colnames[:13]
target = colnames[13]

x=data[predictors]
y=data[target]

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_jobs=2,oob_score=True,n_estimators=10)

forest.fit(x,y)
predi=forest.predict(x)
print(forest.score(x, y))

data['predi'] = predi

data['%error1'] = (data['predi']-data['medv'])**2

print(sum(data['%error1'])/len(data))

data['r_forest']=forest.oob_prediction_

data['%error2'] = (data['r_forest']-data['medv'])**2
print(sum(data['%error2'])/len(data))
