from glob import glob
import os

# 提交樣板路徑
sub_template_path = "data\\public_submission_template.csv"
price_col = "predicted_price"
# 訓練資料路徑
train_path = "data\\training_data.csv"
# 公開測試路徑
public_path = "data\\public_dataset.csv"

# External
external_dir = "data\\external_data"
external_paths = [path for path in glob(external_dir+"\\*.csv")]
external_basenames = [os.path.basename(path) for path in glob(external_dir+"\\*.csv")]

# col setting
symbolic_names = ["使用分區", "主要用途", "主要建材", "建物型態"]
numerical_names = ["土地面積", "移轉層次", "總樓層數", "屋齡", "建物面積", "車位面積", "車位個數", "主建物面積", "陽台面積", "附屬建物面積"]
coordinate_names = ["橫坐標", "縱坐標"]
truth_name = ["單價"]