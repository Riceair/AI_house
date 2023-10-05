import numpy as np

class Normalizer:
    def __init__(self, values, target_min=0, target_max=1):
        values = np.array(values)
        self.target_min = target_min
        self.target_max = target_max
        self.original_min = np.min(values)
        self.original_max = np.max(values)
    
    def normalize(self, values):
        values = np.array(values)
        norm_values = (values - self.original_min) / (self.original_max - self.original_min) *\
                        (self.target_max - self.target_min) + self.target_min
        return norm_values
    
    def denormalize(self, values):
        values = np.array(values)
        denorm_values = (values - self.target_min) / (self.target_max - self.target_min) * \
                            (self.original_max - self.original_min) + self.original_min
        return denorm_values
        
if __name__=="__main__":
    values = [-2.5, -1, 2, 5, 8, 9.9, 10]
    normalizer = Normalizer(values, -1, 1)
    norm_values = normalizer.normalize(values)
    denorm_values = normalizer.denormalize(norm_values)
    print(norm_values)
    print(denorm_values)