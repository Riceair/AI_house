from sklearn.ensemble import RandomForestRegressor
from preprocessors import Preprocessor
from config.data_setting import *
import pandas as pd
import numpy as np
import joblib

# preprocessing
symbolic_type = "onehot"
save_path = "save_models/rf_"+symbolic_type+".pkl"
result_path = "result/rf_"+symbolic_type+".csv"
preprocessor = Preprocessor(symbolic_type=symbolic_type, num_min=0, num_max=1)
X_train = preprocessor.preprocess(train_path)
y_train = preprocessor.y_true
y_train = np.ravel(y_train)

# model = RandomForestRegressor(random_state=0)
# model.fit(X_train, y_train)
# joblib.dump(model, save_path)
model = joblib.load(save_path)

X_test = preprocessor.preprocess(public_path)
y_test = model.predict(X_test)
y_test = preprocessor.denormalizeY(y_test)

df = pd.read_csv(sub_template_path)
df = pd.DataFrame(df)
df[price_col] = y_test
df.to_csv(result_path, index=False)