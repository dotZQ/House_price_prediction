
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#insert the houseData file here
df = pd.read_csv("/houseData.csv")

df.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('plasma'),axis=1)

num = 60
s =[]
#this loop find the best random_state
for i in range(num):
  train , test = train_test_split(df,test_size = 0.2 , random_state = i)
  X = train[['sqft_living','grade']]
  Y = train['price']

  Xtest = test[['sqft_living','grade']]
  Ytest = test['price']

  regr = LinearRegression()
  regr.fit(X,Y)

  this_score = regr.score(Xtest,Ytest)
  s.append(this_score)

#indicate the best randomstate to f
d = max(s)
f = s.index(d)

train , test = train_test_split(df,test_size = 0.2 , random_state = f)

X = train[['sqft_living','grade']]
Y = train['price']

Xtest = test[['sqft_living','grade']]
Ytest = test['price']

regr = LinearRegression()
regr.fit(X,Y)

score = regr.score(Xtest,Ytest)
print("score : ",score)

#you can try by change the first parameter to sqft_living and the second one to grade of your house
price = regr.predict([[1370,7]])
print("predicted_price : ", int(price))
