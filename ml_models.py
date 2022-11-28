import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dataset_skin = pd.read_csv("D:/Projects/Skin_cancer/Skin-Cancer-Detection/final.csv")

transformer = make_column_transformer((OneHotEncoder(), ['target']),remainder='passthrough')
transformed = transformer.fit_transform(dataset_skin)
transformed_df = pd.DataFrame(transformed,columns=transformer.get_feature_names())

# X = dataset_skin.drop(['colorind','target'], axis=1).values
# y = dataset_skin['target'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Classifier = SVC(kernel="linear")

print(transformed_df.head(5))

