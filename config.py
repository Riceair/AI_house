from glob import glob
import os

# 提交樣板路徑
sub_template_path = "data\\public_submission_template.csv"
# 訓練資料路徑
train_path = "data\\training_data.csv"
# 公開測試路徑
public_path = "data\\public_dataset.csv"

# External
external_dir = "data\\external_data"
external_paths = [path for path in glob(external_dir+"\\*.csv")]
external_basenames = [os.path.basename(path) for path in glob(external_dir+"\\*.csv")]