import numpy as np
from torch import nn
import torch
from omegaconf import DictConfig

class DynamicMLP(nn.Module):
    def __init__(self, num_features, first_layer_dim, num_layers, output_dim, batch_norm, dropout_prob=0.5):
        
        layers_dims = self.generate_layer_dims(first_layer_dim, num_layers)
        
        super(DynamicMLP, self).__init__()
        layers = []
        input_dim = num_features

        for previous_layer_output_dim in layers_dims:  # Excluding the output layer
            layers.append(nn.Linear(input_dim, previous_layer_output_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(previous_layer_output_dim))  # Batch normalization
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))  # Dropout
            input_dim = previous_layer_output_dim

        # Add the output layer without BatchNorm and Dropout
        layers.append(nn.Linear(layers_dims[-1], output_dim))

        self.layers = nn.ModuleList(layers)
        
    def generate_layer_dims(self, first_layer_dim, num_layers):
        assert first_layer_dim % 4 == 0, "First layer dimension must be a multiple of 4."
        assert num_layers >= 1, "Number of layers must be at least 1."

        if num_layers == 1:
            return [first_layer_dim]

        last_layer_dim = first_layer_dim // 4
        layer_dims = np.linspace(first_layer_dim, last_layer_dim, num_layers)

        layer_dims = np.round(layer_dims / 4) * 4
        layer_dims = layer_dims.astype(int)  # Convert to integer

        return layer_dims.tolist()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# old models
class Encoder(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig, input_shape: int) -> None:
        super().__init__()

        self.model_cfg = model_cfg

        dims = self._layers_dimensions(input_shape)

        self.model = nn.Sequential()

        if self.model_cfg.name in ("ae", "dae"):
            # shallow network
            if len(dims) <= 2:
                layer = nn.Linear(dims[0], dims[1])
                if self.model_cfg.layer_init == "glorot_uniform":
                    layer.weight = nn.init.xavier_uniform_(layer.weight)
                self.model.add_module(name="layer {i}", module=layer)
            else:
                raise Exception("yet to be implemented")
                for i in range(1, len(dims)):
                    layer = nn.Linear(dims[i-1], dims[i])
                    if self.model_cfg.layer_init == "glorot_uniform":
                        layer.weight = nn.init.xavier_uniform_(layer.weight)
                    self.model.add_module(name="layer {i}", module=layer)
    
    def _layers_dimensions(self, input_shape: int):
        return [input_shape] + self.model_cfg.layers_dims
    
    def _last_activation_function(self, index):
        return (f'sigmoid_{index}', nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class Decoder(Encoder):
    def _layers_dimensions(self, input_shape: int):
        return list(reversed(super()._layers_dimensions(input_shape)))
        
    def _last_activation_function(self, index):
        return (f'relu_{index}', nn.ReLU())


class SimpleAutoEncoder(torch.nn.Module):
    def __init__(self, model_cfg: DictConfig, input_shape: int):
        super().__init__()
        self.encoder = Encoder(model_cfg, input_shape)
        self.decoder = Decoder(model_cfg, input_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))