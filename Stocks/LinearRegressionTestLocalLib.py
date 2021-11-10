# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import math
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from Data.Regression.lib import LinearRegressionLib as lrl
import pickle
# ---------------

style.use('ggplot')

# --- Data Structuring ---

# getting path of data
path = "C:\\Users\\naiza\\PycharmProjects\\MachineLearningAlgorithms\\Data\\Regression\\NY " \
       "Stocks\\AAL Stock 10-16.csv"

# reading path of data into program
df = pd.read_csv(path, header=0, index_col='date', parse_dates=True)

# print(df.head())

# taking only the meaningful features
df = df[['open', 'high', 'low', 'close', 'volume']]
df['HL_PCT'] = (df['high'] - df['close'] / df['close'] * 100.0)
df['PCT_change'] = (df['close'] - df['open'] / df['open'] * 100.0)

# Defining a new data frame
df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]

# print(df.head())

# creating a forecast column
forecast_col = 'close'

# any missing data becomes an outlier
df.fillna(-99999, inplace=True)

# High Level: predicting closing prices in the future
# Low Level: this number will be how far up we shift the label col
forecast_out = int(math.ceil(0.1*len(df)))

# creating a label col as being the close col shifted up by forecast_out
df['label'] = df[forecast_col].shift(-forecast_out)


# print(df.head())
# ------------------------

# --- Training & Testing the Classifier ---

# the x values are our features
X = np.array(df.drop(['label', 'close'], 1))
# standardizes the data-set to a gaussian distribution
X = preprocessing.scale(X)

# X_lately is the forecast_out(18) days omitted by X
X_lately = X[-forecast_out:]
# removing last forecast_out(18) days
X = X[:-forecast_out]


# drop any missing data when going to create the labels
df.dropna(inplace=True)

# the y values are our label
y = np.array(df['label'])

# separating the df into training and testing data
# NOTE: the initialization order matters (X's first, y's second)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# --- all of this has been saved to a pickle so it can be commented out ---

# give the classifier our training data
lrl.fit(X_train, y_train)

# pickling saves the classifier
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(lrl, f)
# -------------------------------------------------------------------------

# opening the pickle and redefining the classifier
pickle_in = open('linearregression.pickle', 'rb')
lrl = pickle.load(pickle_in)

# find the accuracy of the classifier
accuracy = clf.score(X_test, y_test)

# formatted_accuracy = "{:.4}".format(accuracy)
# formatted_accuracy = str(float(formatted_accuracy)*100)
# print("The Linear Regression Classifier was", formatted_accuracy + "%", "accurate with predicting closing stock
# prices", forecast_out, "days in advance.")
# -----------------------------------------

# the prediction made by the linear classifier
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

# creates a col called forecast that is defaulted to all NAN data
df['forecast'] = np.nan

# takes the date of the last row in the data set
last_date = df.iloc[-1].name
# turns that date into UNIX time (Seconds per Epoch)
last_unix = last_date.timestamp()
# the amount of seconds in a day
one_day = 86400
# the next day will be the last day on the data frame + a day's worth of seconds
next_unix = last_unix + one_day

# for each predicted closed price in the next 18 days
for i in forecast_set:
    # formats the next_unix date back to POSIX time (original format)
    next_date = datetime.datetime.fromtimestamp(next_unix)
    # increments the UNIX time by one day
    next_unix += one_day
    # since next_date is not in the data frame this creates a new row at the bottom of the data frame with the
    # respective predicted value from forecast_set
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


# --- Plotting the Data ---

df['close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# close
# plt.scatter(X.T[0], y, c='k')
# plt.show()

# HL_PCT
# plt.scatter(X.T[1], y, c='r')
# plt.show()

# PCT_change
# plt.scatter(X.T[2], y, c='b')
# plt.show()

# volume
# plt.scatter(X.T[3], y, c='g')
# plt.show()
# -------------------------
