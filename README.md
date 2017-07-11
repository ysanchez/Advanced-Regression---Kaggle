# Advanced-Regression-housing-pricing-prediction

This was a kaggle competition https://www.kaggle.com/c/house-prices-advanced-regression-techniques. The goal is to predict the housing price of houses on the market. 

I split the project into two parts. One, was cleaning and engineering the data given. Two, was finding and using the correct algorthm to gave the best outcome. Thsi helped in keeping the project organized and easier to manage.

### Example of cleaining data 

``
import pandas as pd
import numpy as np
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
frames = [train, test]
data = pd.concat(frames)
data['YearRemodAdd'] = data['YearRemodAdd'].apply(lambda x: 2017 - x)
``

### Example of algorithm

```
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5, random_state=10)
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor()
clf = clf.fit(x_train, y_train)
y_predictgrad = clf.predict(x_test)
(This was not the final algorithm used)
```

The final algorithm used was Logistic Regression. The Kaggle project and information provide on their website decribes the data as linear, which is why Logistic Regression was used. 

## Tools
Python 3.0
Jupyter via Anaconda
