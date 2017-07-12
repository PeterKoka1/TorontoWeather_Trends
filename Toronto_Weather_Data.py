import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import Imputer
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
warnings.filterwarnings('ignore')

def toronto_weather_trends():
  data_set = pd.read_csv("C:\\Users\\PeterKokalov\\Desktop\\Data\\TorontoWeatherData.csv")
  data_set = data_set.sort(columns=["Year","Month"])
  data_set.columns = ['Year', 'Month', 'Mean_Temp', 'Max_Temp',
                      'Min_Temp', 'Total_Rain', 'Total_Snow(cm)']
  data_set.Mean_Temp = data_set.Mean_Temp.astype('int')
  data_set.Year = data_set.Year.astype('int')

  #imputer = Imputer(missing_values='NaN', strategy='mean')
  #imputer = imputer.fit(data_set[:, 2])
  #data_set[:, 2] = imputer.fit_transform(data_set[:, 2])

  sample = data_set[data_set["Year"]==1938]
  sample = np.mean(sample["Mean_Temp"])
  #print(sample)

  temperature = []
  for i in data_set["Year"]:
      avg = data_set[data_set["Year"] == i]
      avg = np.mean(avg["Mean_Temp"])
      temperature.append(avg)

  X_opt = pd.DataFrame({'Year': np.array(data_set["Year"]),
                        'Month': np.array(data_set["Month"])})

  y_opt = pd.DataFrame({'Mean_Temp': np.array(temperature),
                      'Year': np.array(data_set["Year"])})
  temperature = pd.DataFrame({"Mean_Temp_2": np.array(temperature)})

  merged_data = pd.merge(left=X_opt, right=y_opt, how='inner', on='Year')

  plot = sns.lmplot(data=merged_data, x='Year', y='Mean_Temp', fit_reg=False)
  plt.title('Toronto Airport Temperature Investigation (1940-2012)')
  plt.xlabel('Year')
  plt.ylabel('Average Temperature')
  plt.figure(figsize=(15,13))
  plt.plot()

  X = merged_data.iloc[:,1:-1]  # vectorizing independent variables
  y = merged_data.iloc[:,2]  # matrix of features

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)

  print(regressor.predict(1940))
  # predicted average temperature for 1940 based on model is 6.89 degrees
  print(regressor.predict(2010))
  # predicted average temperature for 2010 based on model is 8.12 degrees

toronto_weather_trends()
