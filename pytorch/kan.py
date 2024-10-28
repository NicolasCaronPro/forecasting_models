import torch
import torch.nn.functional as F
import math
from forecasting_models.pytorch.utils import *
from forecasting_models.pytorch.models import *
from torch.nn import LSTMCell

######################################################################################################################
#                                                                                                                    #
#                                           Implementation of                                                        #
#                                    https://github.com/Blealtan/efficient-kan                                       #                              #
#                                                                                                                    #
######################################################################################################################
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        base_activation,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        #print(x.shape, self.in_features)
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        layers_hidden,
        end_channels,
        k_days,
        device,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        act_func='silu',
        grid_eps=0.02,
        grid_range=[-1, 1],
        binary=False,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        if act_func == 'gelu':
            self.base_activation = torch.nn.GELU
        elif act_func == 'relu':
            self.base_activation = torch.nn.ReLU
        elif act_func == 'silu':
            self.base_activation = torch.nn.SiLU

        #self.input = torch.nn.Conv1d(in_channels=in_channels, out_channels=layers_hidden[0], kernel_size=1).to(device)
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=self.base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

        self.output = OutputLayer(in_channels=layers_hidden[-1],
                                  end_channels=end_channels,
                                  n_steps=k_days,
                                  device=device, act_func=act_func,
                                  binary=binary)
        
        self.calibrator_layer = torch.nn.Conv1d(
            in_channels=layers_hidden[0] + 1,
            out_channels=1,
            kernel_size=1,
            bias=True,
            device=device
        )
        
        """self.calibrator_layer = GCNConv(
            in_channels=layers_hidden[0] + (2 if binary else 1),
            out_channels=2 if binary else 1,
            bias=True,
            # Additional parameters can be added if needed
        )"""

    def forward(self, x: torch.Tensor, edge_index, update_grid=False):
        
        #x = self.input(x)
        
        if len(x.shape) == 3:
            x = x[:, :, -1]
        
        original_input = x

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        x = self.output(x)
        
        """x = torch.concat((original_input, x), dim=1)
        x = self.calibrator_layer(x, edge_index)
        x = torch.clamp(x, min=0)"""
              
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
        
###############################################################################################################################
#                                                Temporal KAN                                                                 #
#                                                                                                                             #
#                                          https://github.com/remigenet/TKAN/blob/main/                                       #
#                                                                                                                             #
#                                                                                                                             #
###############################################################################################################################


class TKANCELL(torch.nn.Module):
    def __init__(self, input_size, hidden_size, base_activation, kan_config):
        super(TKANCELL, self).__init__()

        self.weight_ih = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.weight_hh = torch.nn.Parameter(torch.empty((4 * hidden_size, hidden_size)))
        self.weight_ig = torch.nn.Parameter(torch.empty((4 * hidden_size, hidden_size)))
        self.weight_io = torch.nn.Parameter(torch.empty((4 * hidden_size, hidden_size)))
        
        self.b_ih = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.b_hh = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.b_ig = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        self.b_io = torch.nn.Parameter(torch.empty((4 * hidden_size, input_size)))
        
        self.hidden_size = hidden_size
        self.kanlayer = KANLinear(in_features=input_size,
                                  out_features=hidden_size,
                                  base_activation=base_activation, **kan_config)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialisation selon Glorot pour les poids
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.xavier_uniform_(self.weight_hh)
        torch.nn.init.xavier_uniform_(self.weight_ig)
        torch.nn.init.xavier_uniform_(self.weight_io)
        
        # Initialisation des biais à zéro
        torch.nn.init.zeros_(self.b_ih)
        torch.nn.init.zeros_(self.b_hh)
        torch.nn.init.zeros_(self.b_ig)
        torch.nn.init.zeros_(self.b_io)
        
        # Initialisation des paramètres de la couche KAN
        self.kanlayer.reset_parameters()

    def forward(self, X, hx):
        if hx is None:
            hx = torch.zeros(X.size(0), self.hidden_size, device=X.device)
            cx = torch.zeros(X.size(0), self.hidden_size, device=X.device)
        else:
            hx, cx = hx

        forgetgate = F.linear(X, self.weight_ih, self.b_ih)
        ingate = F.linear(hx, self.weight_hh, self.b_hh)
        kan_input = F.linear(X, self.weight_io, self.b_io)
        outgate = self.kanlayer(kan_input)
        cellgate = F.linear(cx, self.weight_ig, self.b_ig)
        
        ingate = torch.nn.Sigmoid(ingate)
        forgetgate = torch.nn.Sigmoid(forgetgate)
        cellgate = torch.nn.Tanh(cellgate)
        outgate = torch.nn.Sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy
    
class TKAN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, end_channels, act_func, dropout, binary, k_days, return_hidden, device, kan_config):
        super(TKAN, self).__init__()

        self.return_hidden = return_hidden
        if act_func == 'gelu':
            base_activation = torch.nn.GELU
        elif act_func == 'relu':
            base_activation = torch.nn.ReLU
        elif act_func == 'silu':
            base_activation = torch.nn.SiLU

        self.hidden_channels = hidden_size[0]

        self.input = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size[0], kernel_size=1)
        self.num_tkan_layer = len(hidden_size)
        self.tkan_layers = torch.nn.ModuleList()
        for i in range(self.num_tkan_layer - 1):
            tkan = TKANCELL(input_size=hidden_size[0], hidden_size=hidden_size[i+1], base_activation=base_activation, kan_config=kan_config)
            self.tkan_layers.append(tkan)

        self.output = OutputLayer(in_channels=hidden_size[-1], end_channels=end_channels,
                                  n_steps=k_days, device=device, act_func=act_func,
                                  binary=binary)
        
        self.batchNorm = nn.BatchNorm(hidden_size[-1]).to(device)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, X, edges):

        batch_size = X.size(0)
        h0 = torch.zeros(self.num_tkan_layer, batch_size, self.hidden_channels, device=X.device)
        c0 = torch.zeros(self.num_tkan_layer, batch_size, self.hidden_channels, device=X.device)
        x = self.input(X)
        x = torch.movedim(x, 2, 1)

        for layer in self.tkan_layers:
            h0, c0 = layer(x, (h0, c0))

        x = torch.squeeze(x[:, -1, :])
        x = self.batchNorm(x)
        x = self.dropout(x)
        output = self.output(x)

        if self.return_hidden:
            return output, x
        else:
            return output