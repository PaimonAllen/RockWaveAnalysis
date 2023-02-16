import tsfel
import pandas as pd
# import numpy as np

# load dataset
df = pd.read_csv(
    '/home/tzr/DataLinux/Documents/GitHubSYNC/RockWaveAnalysis\
/1.shapeClassification/KNN/dataset/processAndTemp/1.shapeTotal/1-1')

# Retrieves a pre-defined feature configuration file to extract all available features
cfg = tsfel.get_features_by_domain()

# Extract features
X = tsfel.time_series_features_extractor(cfg, df, fs=20000)
print(X)
# print(X.shape)
# X1 = np.array(X)
# print(X1)
# T = pd.DataFrame(X, columns=X.index, index=X.columns)
# print(T)
# T.to_csv('./test1.csv')
column_indexs = []
for column_index, row_data in X.iteritems():
    column_indexs.append(column_index)
    print(row_data)
print(column_indexs)

