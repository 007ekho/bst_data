# Importing the required packages
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn import metrics
# import joblib

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# load the dataset
dataset = pd.read_csv("train.csv")
print(dataset)





# Train test split
X = dataset.drop(columns=['end_stop'])
y = dataset['end_stop']
RANDOM_SEED = 6

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)

# RandomForest
rfr = RandomForestClassifier(random_state=RANDOM_SEED)

model_forest = rfr.fit(X_train, y_train)

joblib.dump(model_forest, 'pred_model.joblib')

loaded_model = joblib.load('pred_model.joblib')

data = [[
                1000010,
                20,
                4,
                6
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")