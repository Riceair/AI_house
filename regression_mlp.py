from preprocessors import Preprocessor
from config.data_setting import *
from models.mlp import MultiMLP
import pandas as pd
import numpy as np
import torch

# preprocessing
hidden_num = 1
save_path = "save_models/mlp_h"+str(hidden_num)+".pt"
result_path = "result/mlp_h"+str(hidden_num)+".csv"
preprocessor = Preprocessor(symbolic_type="onehot", num_min=0, num_max=1, y_min=0, y_max=1)
X_train = preprocessor.preprocess(train_path)
y_train = preprocessor.y_true
# y_train = np.ravel(y_train)

# model = RandomForestRegressor(random_state=0)
# model.fit(X_train, y_train)
# joblib.dump(model, save_path)

# X_test = preprocessor.preprocess(public_path)
# y_test = model.predict(X_test)
# y_test = preprocessor.denormalizeY(y_test)

# df = pd.read_csv(sub_template_path)
# df = pd.DataFrame(df)
# df[price_col] = y_test
# df.to_csv(result_path, index=False)