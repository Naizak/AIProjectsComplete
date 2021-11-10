import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections


mm_data = pd.read_csv('Mega Millions.csv')
megaball = mm_data.mb
pb_data = pd.read_csv('Powerball Winning.csv')
powerball = pb_data.pb
legend = ['First Number', 'Second Number']

"""
num1sorted = mm_data['num1'].value_counts()
num1 = mm_data['num1'].values
"""

mm_data.T.plot(kind="bar", legend=False)
plt.show()



"""
plt.bar(num1freq[0], num1freq[1])
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.legend(legend)
plt.xticks(range(1, 70))
plt.title('Mega Millions Number Frequency')
plt.show()
"""



