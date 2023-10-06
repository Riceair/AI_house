import torch
import torch.nn as nn

class MultiMLP(nn.Module):
    def __init__(self, input_dim : int, hidden_dims : list[int], output_dim : int, out_selection=None):
        '''
        input_dim: input dimension \n
        hidden_dims: dimensions of hidden layers \n
        output_dim: output dimension \n
        out_selection: output activation funciont -> 'sigmoid', 'relu', 'tanh' \n
        '''
        super(MultiMLP, self).__init__()
        layers = []
        prev_dim = input_dim

        # 動態隱藏層
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        self.layers = nn.Sequential(*layers)

        # 輸出層
        self.output_layer = nn.Linear(prev_dim, output_dim)
        # 輸出層激發函數
        if out_selection == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif out_selection == 'relu':
            self.output_activation = nn.ReLU()
        elif out_selection == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = None
    
    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        if self.output_activation != None:
            x = self.output_activation(x)
        return x
    
if __name__ == "__main__":
    input_dim = 40
    hidden_dims = [128, 128,]
    output_dim = 1
    model = MultiMLP(input_dim, hidden_dims, output_dim, 'tanh')
    print(model)