############################## TGAT #################################
    
# Code implementation of TGAT model following torch Geometric API propose by Da Xu et al. See https://arxiv.org/abs/2002.07962
# Code origin https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, tmax, device, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        self.time_dim = tmax
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(((1 / 10 ** torch.linspace(0, tmax, expand_dim, device=device, dtype=torch.float32))))
        self.phase = torch.nn.Parameter(torch.zeros(expand_dim, dtype=torch.float32, device=device))

        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)
        
    def forward(self, ts):
        # ts: [N, L
        batch_size = ts.size(0)
        seq_len = ts.size(1)
    
        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]

        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        #map_ts = torch.masked_select(map_ts, map_ts.gt(0))
        harmonic = torch.cos(map_ts)
     
        return harmonic #self.dense(harmonic)
    
class TGATCONV(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, torch.Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = torch.nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = torch.nn.Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = torch.nn.Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(  # noqa: F811
        self,
        x: torch.Tensor,
        edge_index,
        edge_harmonic,
        edge_attr = None,
        size = None,
        return_attention_weights: Optional[bool] = None,
    ):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, edge_harmonic=edge_harmonic,
                                  size=size)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(self, alpha_j: torch.Tensor, alpha_i: torch.Tensor,
                    edge_harmonic : torch.Tensor,
                    edge_attr: torch.Tensor, index: torch.Tensor, ptr: torch.Tensor,
                    dim_size: Optional[int]) -> torch.Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_harmonic is not None:
            alpha = alpha + edge_harmonic

        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
def timedelta(n_sequence):
    res = torch.zeros(n_sequence)
    res -= torch.abs(n_sequence)
    return res

class TGAT(torch.nn.Module):
    def __init__(self, n_sequences, in_channels_list, heads, bias, dropout, device):
        super(TGAT, self).__init__()
        self.n_sequences = n_sequences
        self.device = device
        self.layers = []
        heads = [1] + heads
        num_of_layers = len(in_channels_list)
        in_channels_list = in_channels_list + [heads[-1] * in_channels_list[-1]]
        self.time_encoder = TimeEncode(expand_dim=1, tmax=n_sequences, device=device)
 
        for i in range(num_of_layers):
            layer = TGATCONV(in_channels=in_channels_list[i] * heads[i],
                            out_channels=in_channels_list[i + 1],
                            heads=heads[i],
                            concat = False if i < num_of_layers - 1 else False,
                            negative_slope=0.2,
                            fill_value='mean',
                            edge_dim=None,
                            add_self_loops=True,
                            dropout=dropout, bias=bias).to(device)

            self.layers.append(layer)

        self.gelu = GELU()

        self.fc = nn.Linear(in_channels=in_channels_list[-1], out_channels=in_channels_list[-1])
        self.relu = ReLU()
        self.fc2 = nn.Linear(in_channels=in_channels_list[-1], out_channels=1)

    def forward(self, X, edge_index):
        x = X
        edge_time = edge_index[-1]
        edge_index = edge_index[:2]
        edge_time = edge_time.view(-1, 1)
        edge_harmonic = self.time_encoder(edge_time)
        edge_harmonic = edge_harmonic.view(-1, 1)
        #edge_harmonic = None
        for layer in self.layers:
            x = layer(x, edge_index, edge_harmonic)
            x = self.gelu(x)

        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(X.shape[0])
        return x