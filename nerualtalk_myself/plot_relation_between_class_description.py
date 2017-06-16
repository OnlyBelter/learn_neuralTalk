
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv(r'images_class_and_description.csv')
training_data_num = 10

data_X = np.array(data.class_P)
data_Y = np.array(np.exp(data.des_LogP))*100
point_tag = data['Image Id']

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(data_X, data_Y)
for i, name in enumerate(point_tag):
    ax.annotate(name, (data_X[i], data_Y[i]))
coor = stats.pearsonr(data_X, data_Y)
print(type(coor))
plt.xlabel("Class1 prob")
plt.ylabel("Desc e^logP*100")
plt.annotate('Pearson correlation: ' + str(round(coor[0], 5)), xy=(0.1, 0.85))

plt.show()