# Import models and utility functions

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import datasets
  

SEED = 1
  
#Load any dataset from sklearn that you would like to load.
#I've chosen the diabetes dataset.
diabetes = datasets.load_diabetes()

#Set X and y to the data and the target
X, y = diabetes.data, diabetes.target
  
#Split the dataset. Common ML practice.
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = SEED)

#create instance of GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators = 200, max_depth = 1, random_state = SEED)
  
#fit (another common/essential ML practice)
gbr.fit(train_X, train_y)
  
#declare prediction for dataset
pred_y = gbr.predict(test_X)
  
#create rmse (mean squared error) value, then print
test_rmse = MSE(test_y, pred_y) ** (1 / 2)
print('RMSE test set: {:.2f}'.format(test_rmse))