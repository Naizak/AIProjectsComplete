import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


path = 'C:\\Users\\naiza\\PycharmProjects\\MachineLearningAlgorithms\\Data\Regression' \
       '\\Communities and Crime\\communities.csv'

df = pd.read_csv(path)

print(df.head())

# df = df[['open', 'high', 'low', 'close', 'volume']]

# plt.scatter(xs, ys)
# plt.scatter(predict_x, predict_y, s=100, color='g')
# plt.plot(xs, regression_line)
# plt.show()
