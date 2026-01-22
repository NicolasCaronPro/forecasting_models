
"""class MultiScaleGraph(torch.nn.Module):
    def __init__(self, input_channels, graph_input_channels, graph_output_channels, device, graph_or_node, task_type,
                 num_output_scale=1, num_sequence=1, out_channels=5):
        
        super(MultiScaleGraph, self).__init__()
        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device=device

        ### Embedding Layer
        self.embedding = torch.nn.Linear(input_channels * num_sequence, graph_input_channels).to(device)

        ### One GCN per scale
        self.gcn_layers = torch.nn.ModuleList([
            GraphConv(graph_input_channels, graph_output_channels).to(device) for _ in range(num_output_scale)
        ])

        ### Encoder: upscale to next scale (Linear for feature transfer)
        self.encoder = torch.nn.Linear(graph_input_channels, graph_input_channels).to(device)

        ### Decoder: downscale from higher scale
        self.decoder = torch.nn.Linear(graph_output_channels, graph_output_channels).to(device)

        ### Output layers (per scale)
        self.output_heads = torch.nn.ModuleList([
            nn.Linear(graph_output_channels * 2, out_channels).to(device) for _ in range(num_output_scale)
        ])

    def forward(self, X, graph_scale_list: list, increase_scale: list, decrease_scale: list):

        #print(graph_scale_list)
        #print(decrease_scale)
        #print(increase_scale)
        assert self.num_output_scale == len(graph_scale_list)
        batch_size = X.shape[0]
        
        X = X.view(batch_size, -1)

        ### Step 1: Embed input features at scale 0
        features_per_scale = [self.embedding(X)]  # scale 0 only

        ### Step 2: Encode to upper scales using increase_scale
        for i in range(self.num_output_scale - 1):
            g = increase_scale[i].to(X.device)
            src_feat = features_per_scale[i]
            g.srcdata["h"] = self.encoder(src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            features_per_scale.append(dst_feat)

        ### Step 3: Apply GCN at each scale
        gcn_outputs = []
        for i in range(self.num_output_scale):
            g = graph_scale_list[i].to(X.device)
            h = self.gcn_layers[i](g, features_per_scale[i])
            gcn_outputs.append(h)

        ### Step 4: Decode from top to bottom using decrease_scale

        decoded_features = [gcn_outputs[-1]]  # start from top
        for i in range(self.num_output_scale - 1):
            g = decrease_scale[i].to(X.device)
            src_feat = decoded_features[0]
            g.srcdata["h"] = self.decoder(src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            decoded_features.insert(0, dst_feat)

        ### Step 5: Final prediction per scale
        outputs = []
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            outputs.append(F.softmax(logits, dim=-1))

        outputs = torch.cat(outputs, dim=0)
        return outputs"""

class MultiScaleGraph(torch.nn.Module):
    def __init__(
        self, input_channels, features_per_scale, device, num_output_scale,
        graph_or_node, task_type, num_sequence=1, out_channels=5, return_hidden=False, horizon=0
    ):
        super(MultiScaleGraph, self).__init__()

        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device = device
        self.features_per_scale = features_per_scale
        self.return_hidden = return_hidden
        self.horizon = horizon
        self.n_sequences = num_sequence
        self.end_channels = features_per_scale[-1]

        ### Embedding Layer (for scale 0 only)
        self.embedding = nn.Linear(input_channels * num_sequence, features_per_scale[0]).to(device)

        ### GCN layers (one per scale)
        self.gcn_layers = torch.nn.ModuleList([
            GraphConv(features_per_scale[i], features_per_scale[i]).to(device)
            for i in range(self.num_output_scale)
        ])

        ### Encoder: separate encoder per scale (except top scale)
        self.encoders = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i], features_per_scale[i + 1]).to(device)
            for i in range(self.num_output_scale - 1)
        ])

        ### Decoder: separate decoder per scale (except top scale)
        self.decoders = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i + 1], features_per_scale[i]).to(device)
            for i in reversed(range(self.num_output_scale - 1))
        ])

        ### Output heads: one per scale
        self.output_heads = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i] * 2, out_channels).to(device)
            for i in range(self.num_output_scale)
        ])

    def forward(self, X, graph_scale_list: list, increase_scale: list, decrease_scale: list, z_prev=None):
        assert self.num_output_scale == len(graph_scale_list)
        batch_size = X.shape[0]

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        X = X.view(batch_size, -1)

        ### Step 1: Embed input at scale 0
        features_per_scale = [self.embedding(X)]

        ### Step 2: Encode to higher scales
        for i, g in enumerate(increase_scale):
            g = g.to(X.device)
            src_feat = features_per_scale[i]
            g.srcdata["h"] = self.encoders[i](src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            features_per_scale.append(dst_feat)

        ### Step 3: GCN at each scale
        gcn_outputs = []
        for i, g in enumerate(graph_scale_list):
            g = g.to(X.device)
            h = self.gcn_layers[i](g, features_per_scale[i])
            gcn_outputs.append(h)

        ### Step 4: Decode from top to bottom
        decoded_features = [gcn_outputs[-1]]
        for i, g in zip(reversed(range(self.num_output_scale - 1)), decrease_scale):
            g = g.to(X.device)
            src_feat = decoded_features[0]
            g.srcdata["h"] = self.decoders[self.num_output_scale - 2 - i](src_feat)
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "h_dst"))
            dst_feat = g.dstdata["h_dst"]
            decoded_features.insert(0, dst_feat)

        ### Step 5: Final prediction at each scale
        outputs = []
        logits_list = []
        hidden = None
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            logits_list.append(logits)
            outputs.append(F.softmax(logits, dim=-1))
            if i == self.num_output_scale - 1:
                hidden = combined

        outputs = torch.cat(outputs, dim=0)
        logits = torch.cat(logits_list, dim=0)
        return outputs, logits, hidden

class CrossScaleAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(CrossScaleAttention, self).__init__()
        self.query_proj = nn.Linear(output_dim, output_dim)
        self.key_proj = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)
        self.num_heads = num_heads
        self.scale = (output_dim // num_heads) ** 0.5

    def forward(self, g, src_feat, dst_feat):
        Q = self.query_proj(dst_feat)  # (N_dst, d)
        K = self.key_proj(src_feat)    # (N_src, d)
        V = self.value_proj(src_feat)  # (N_src, d)

        g.srcdata["K"] = K
        g.srcdata["V"] = V
        g.dstdata["Q"] = Q

        def message_func(edges):
            score = (edges.dst["Q"] * edges.src["K"]).sum(dim=-1) / self.scale
            return {"score": score, "V": edges.src["V"]}

        def reduce_func(nodes):
            attn = F.softmax(nodes.mailbox["score"], dim=1)  # (N_dst, num_neighbors)
            V = nodes.mailbox["V"]  # (N_dst, num_neighbors, d)
            out = (attn.unsqueeze(-1) * V).sum(dim=1)  # (N_dst, d)
            return {"h_dst": out}

        g.apply_edges(message_func)
        g.update_all(message_func, reduce_func)
        return g.dstdata["h_dst"]


class MultiScaleAttentionGraph(torch.nn.Module):
    def __init__(
        self, input_channels, features_per_scale, device, num_output_scale,
        graph_or_node='graph', task_type='classification',
        num_sequence=1, out_channels=5, num_heads=4, return_hidden=False, horizon=0
    ):
        super(MultiScaleAttentionGraph, self).__init__()

        self.num_output_scale = num_output_scale
        self.out_channels = out_channels
        self.is_graph_or_node = graph_or_node == 'graph'
        self.task_type = task_type
        self.device = device
        self.features_per_scale = features_per_scale
        self.return_hidden = return_hidden
        self.horizon = horizon

        # Embedding Layer for scale 0
        self.embedding = nn.Linear(input_channels * num_sequence, features_per_scale[0]).to(device)

        # GCN layers per scale
        self.gcn_layers = torch.nn.ModuleList([
            GraphConv(features_per_scale[i], features_per_scale[i]).to(device)
            for i in range(self.num_output_scale)
        ])

        # Attention Encoders between scales
        self.encoder_attn_blocks = torch.nn.ModuleList([
            CrossScaleAttention(features_per_scale[i], features_per_scale[i + 1], num_heads).to(device)
            for i in range(self.num_output_scale - 1)
        ])

        # Attention Decoders between scales (reverse order)
        self.decoder_attn_blocks = torch.nn.ModuleList([
            CrossScaleAttention(features_per_scale[i + 1], features_per_scale[i], num_heads).to(device)
            for i in reversed(range(self.num_output_scale - 1))
        ])

        # Output layers per scale
        self.output_heads = torch.nn.ModuleList([
            nn.Linear(features_per_scale[i] * 2, out_channels).to(device)
            for i in range(self.num_output_scale)
        ])

    def forward(self, X, graph_scale_list, increase_scale, decrease_scale, z_prev=None):
        assert self.num_output_scale == len(graph_scale_list)
        batch_size = X.shape[0]

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)

        X = X.view(batch_size, -1)

        # Step 1: Initial embedding (scale 0)
        features_per_scale = [self.embedding(X)]

        # Step 2: Encoding to upper scales with attention
        for i, g in enumerate(increase_scale):
            g = g.to(X.device)
            src_feat = features_per_scale[i]
            dst_feat = torch.zeros(g.num_dst_nodes(), self.features_per_scale[i + 1], device=X.device)
            encoded = self.encoder_attn_blocks[i](g, src_feat, dst_feat)
            features_per_scale.append(encoded)

        # Step 3: GCN on each scale
        gcn_outputs = []
        for i, g in enumerate(graph_scale_list):
            g = g.to(X.device)
            h = self.gcn_layers[i](g, features_per_scale[i])
            gcn_outputs.append(h)

        # Step 4: Decoding back down with attention
        decoded_features = [gcn_outputs[-1]]
        for i, g in zip(reversed(range(self.num_output_scale - 1)), decrease_scale):
            g = g.to(X.device)
            src_feat = decoded_features[0]
            dst_feat = torch.zeros(g.num_dst_nodes(), self.features_per_scale[i], device=X.device)
            decoded = self.decoder_attn_blocks[self.num_output_scale - 2 - i](g, src_feat, dst_feat)
            decoded_features.insert(0, decoded)

        # Step 5: Output heads for predictions
        outputs = []
        logits_list = []
        hidden = None
        for i in range(self.num_output_scale):
            combined = torch.cat([gcn_outputs[i], decoded_features[i]], dim=-1)
            logits = self.output_heads[i](combined)
            logits_list.append(logits)
            outputs.append(F.softmax(logits, dim=-1))
            if i == self.num_output_scale - 1:
                hidden = combined

        outputs = torch.cat(outputs, dim=0)
        logits = torch.cat(logits_list, dim=0)
        return outputs, logits, hidden
    
##############################################################################################################
#                                                                                                            #
#                                       TRANSFORMER                                                          #
#                                                                                                            #
##############################################################################################################

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int = 256, dropout: float = 0.1, max_len: int = 30):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
################################################# TransformerNet #############################################################""

class TransformerNet(torch.nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
            self,
            seq_len=30,
            input_dim=24,
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            num_layers=4,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1,
            channel_attention=False,
            graph_or_node='node',
            out_channels=2,
            task_type='binary',
            return_hidden=False,
            horizon=0,
    ):

        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.graph_or_node = graph_or_node == 'graph'
        self.horizon = horizon

        # self.emb = torch.nn.Embedding(input_dim, d_model)
        self.channel_attention = channel_attention
        
        self.lin_time = torch.nn.Linear(input_dim, d_model)
        self.lin_channel = torch.nn.Linear(seq_len, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_time = torch.nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=num_layers,
        )

        encoder_layer_channel = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_channel = torch.nn.TransformerEncoder(
            encoder_layer_channel,
            num_layers=num_layers,
        )

        self.out_time = torch.nn.Linear(d_model, d_model)
        self.out_channel = torch.nn.Linear(d_model, d_model)

        self.lin = torch.nn.Linear(d_model * 2, 2)

        if self.channel_attention:
            self.classifier = torch.nn.Linear(d_model * 2, out_channels)
        else:
            self.classifier = torch.nn.Linear(d_model, out_channels)

        self.d_model = d_model
        self.return_hidden = return_hidden
        self.end_channels = d_model
        self.n_sequences = seq_len

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()  # For regression or custom handling

    def resh(self, x, y):
        return x.unsqueeze(1).expand(y.size(0), -1)

    def forward(self, x_, edge_index=None, z_prev=None):
        if z_prev is None:
            z_prev = torch.zeros((x_.shape[0], self.end_channels, self.n_sequences), device=x_.device, dtype=x_.dtype)

        x_ = x_.permute(2, 0, 1)

        x = torch.tanh(self.lin_time(x_))
        x = self.pos_encoder(x)
        x = self.transformer_encoder_time(x)
        x = x[0, :, :]

        if self.channel_attention:
            y = torch.transpose(x_, 0, 2)
            y = torch.tanh(self.lin_channel(y))
            y = self.transformer_encoder_channel(y)

            x = torch.tanh(self.out_time(x))
            y = torch.tanh(self.out_channel(y[0, :, :]))

            h = self.lin(torch.cat([x, y], dim=1))

            m = torch.nn.Softmax(dim=1)
            g = m(h)

            g1 = g[:, 0]
            g2 = g[:, 1]

            x = torch.cat([self.resh(g1, x) * x, self.resh(g2, x) * y], dim=1)

        hidden = x
        logits = self.classifier(hidden)
        output = self.output_activation(logits)

        return output, logits, hidden

class TransformerNetCutClient(torch.nn.Module):
    def __init__(
            self,
            seq_len=30,
            input_dim=24,
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            num_layers=4,
            dropout=0.1,
            graph_or_node='node',
            horizon=0):

        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.graph_or_node = graph_or_node == 'graph'
        self.horizon = horizon

        # self.emb = torch.nn.Embedding(input_dim, d_model)

        self.lin_time = torch.nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_time = torch.nn.TransformerEncoder(
            encoder_layer_time,
            num_layers=num_layers,
        )

        self.end_channels = d_model
        self.n_sequences = seq_len

    def resh(self, x, y):
        return x.unsqueeze(1).expand(y.size(0), -1)

    def forward(self, x_, edge_index=None, z_prev=None):
        if z_prev is None:
            z_prev = torch.zeros((x_.shape[0], self.end_channels, self.n_sequences), device=x_.device, dtype=x_.dtype)

        x_ = x_.permute(2, 0, 1)
        x = torch.tanh(self.lin_time(x_))
        hidden = x
        logits = x
        output = x
        return output, logits, hidden

class TransformerNetCutServer(torch.nn.Module):

    def __init__(self,
            d_model=256,
            seq_len=1,
            nhead=8,
            dim_feedforward=512,
            num_layers=4,
            dropout=0.1,
            channel_attention=True,
            task_type='binary',
            out_channels=2,
            return_hidden=False,
            graph_or_node='node',
            horizon=0):

        super().__init__()

        self.lin_channel = torch.nn.Linear(seq_len, d_model)
        self.channel_attention = channel_attention

        self.graph_or_node = graph_or_node == 'graph'
        self.horizon = horizon
        
        encoder_layer_channel = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder_channel = torch.nn.TransformerEncoder(
            encoder_layer_channel,
            num_layers=num_layers,
        )

        self.out_time = torch.nn.Linear(d_model, d_model)
        self.out_channel = torch.nn.Linear(d_model, d_model)
        #self.lin_channel = torch.nn.Linear(seq_len, d_model)

        self.lin = torch.nn.Linear(d_model, 2)

        if self.channel_attention:
            self.classifier = torch.nn.Linear(d_model * 2, out_channels)
        else:
            self.classifier = torch.nn.Linear(d_model, out_channels)

        self.d_model = d_model
        self.return_hidden = return_hidden
        self.end_channels = d_model
        self.n_sequences = seq_len

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()  # For regression or custom handling

    def forward(self, x_, z_prev=None):
        if z_prev is None:
            z_prev = torch.zeros((x_.shape[0], self.end_channels, self.n_sequences), device=x_.device, dtype=x_.dtype)

        if self.channel_attention:
            y = torch.transpose(x_, 0, 2)

            y = torch.tanh(self.lin_channel(y))
            
            y = self.transformer_encoder_channel(y)

            y = torch.tanh(self.out_channel(y[0, :, :]))

            h = self.lin(y, dim=1)

            """m = torch.nn.Softmax(dim=1)
            g = m(h)

            g1 = g[:, 0]
            g2 = g[:, 1]

            x = torch.cat([self.resh(g1, x) * x, self.resh(g2, x) * y], dim=1)"""

            x = h
        else:
            x = x_
            
        hidden = x
        logits = self.classifier(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden
        
################################### BAYESIAN ######################################
class BayesianMLP(torch.nn.Module):
    """Minimal Bayesian MLP implemented with blitz."""
    def __init__(self, in_dim, hidden_dim, out_channels, task_type='regression',
                 device='cpu', graph_or_node='node', return_hidden=False, horizon=0):
        super().__init__()
        self.fc1 = BayesianLinear(in_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, out_channels)
        self.to(device)

        self.graph_or_node = graph_or_node
        self.return_hidden = return_hidden
        self.horizon = horizon
        self.end_channels = hidden_dim
        self.n_sequences = 1

        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid()
        else:
            self.output_activation = torch.nn.Identity()

    def forward(self, x, edge_index=None, z_prev=None):
        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)

        x = x[:, :, -1]
        hidden = torch.relu(self.fc1(x))
        logits = self.fc2(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

    def kl_loss(self):
        return kl_divergence_from_nn(self)

class ClassicESN(torch.nn.Module):
    """
    ESN (ReservoirPy) + readout PyTorch.

    Entrée:  x  de forme (B, X, T)
    Sortie: logits (B, n_classes)

    Note: le réservoir est fixe (pas de gradients). Seul le readout PyTorch est entraîné.
    """
    def __init__(
        self,
        in_channels: int,          # X
        out_channels: int = 5,
        # Réservoir
        reservoir_size: int = 300,
        sr: float = 0.9,
        lr_leak: float = 0.2,
        input_scaling: float = 1.0,
        seed: int = 42,
        # Agrégation
        agg: str = "mean_std_last",  # "mean" | "last" | "mean_std_last"
        # Readout (MLP)
        hidden_channels: int = 256,
        dropout: float = 0.2,
        # Normalisation des features réservoir (optionnel)
        use_feat_norm: bool = False,
        eps: float = 1e-8,
        device='cpu',
        task_type='classification',
        horizon: int = 0,
        end_channels: int = 64,
        n_sequences: int = 1,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_channels
        self.n_classes = out_channels
        self.agg = agg
        self.use_feat_norm = use_feat_norm
        self.eps = eps
        self.device = device
        self.horizon = horizon
        self.end_channels = end_channels
        self.n_sequences = n_sequences

        # --- Réservoir ReservoirPy (fixe) ---
        self.reservoir = Reservoir(
            units=reservoir_size,
            sr=sr,
            lr=lr_leak,
            input_scaling=input_scaling,
            seed=seed,
        )

        # Dimension des features après agrégation
        if agg == "last" or agg == "mean":
            feat_dim = reservoir_size
        elif agg == "mean_std_last":
            feat_dim = 3 * reservoir_size
        else:
            raise ValueError("agg doit être 'last', 'mean' ou 'mean_std_last'")

        self.feat_dim = feat_dim

        # Buffers pour normalisation (remplis par fit_feature_norm)
        self.register_buffer("feat_mu", torch.zeros(1, feat_dim))
        self.register_buffer("feat_sigma", torch.ones(1, feat_dim))

        # --- Readout PyTorch (entraînable) ---
        self.layer_norm = torch.nn.LayerNorm(feat_dim)
        self.linear1 = torch.nn.Linear(feat_dim, hidden_channels)
        self.relu = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_channels, out_channels)

        if task_type == 'classification':
            self.output_activation = torch.torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.torch.nn.Softmax(dim=-1).to(device)
        else:
            self.output_activation = torch.torch.nn.Identity().to(device)

    def _seq_to_feature_np(self, seq_TX: np.ndarray) -> np.ndarray:
        """
        seq_TX: (T, X) en numpy float32
        retourne: (feat_dim,)
        """
        states = self.reservoir.run(seq_TX)  # (T, units) numpy

        if self.agg == "last":
            feat = states[-1]
        elif self.agg == "mean":
            feat = states.mean(axis=0)
        elif self.agg == "mean_std_last":
            feat = np.concatenate([states.mean(axis=0), states.std(axis=0), states[-1]], axis=0)
        else:
            raise RuntimeError("agg invalide")

        return feat.astype(np.float32)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, X, T) torch
        retourne: (B, feat_dim) torch (sur le même device que x)
        """
        if x.ndim != 3:
            raise ValueError("x doit être de forme (B, X, T)")
        B, X, T = x.shape
        # if X != self.in_features:
        #     raise ValueError(f"in_features mismatch: attendu X={self.in_features}, reçu X={X}")

        device = x.device

        # IMPORTANT: on passe en CPU/NumPy pour ReservoirPy
        x_np = x.detach().to("cpu").numpy().astype(np.float32)  # (B, X, T)

        feats = []
        for b in range(B):
            seq_TX = x_np[b].T  # (T, X)
            feats.append(self._seq_to_feature_np(seq_TX))

        feats = np.stack(feats, axis=0)  # (B, feat_dim)
        feats_t = torch.from_numpy(feats).to(device=device, dtype=torch.float32)
        return feats_t

    @torch.no_grad()
    def fit_feature_norm(self, x: torch.Tensor, batch_size: int = 64):
        """
        Calcule mu/sigma des features réservoir sur un jeu (x) pour normaliser ensuite.
        x: (B, X, T)
        """
        if not self.use_feat_norm:
            self.feat_mu.zero_()
            self.feat_sigma.fill_(1.0)
            return

        B = x.shape[0]
        all_feats = []
        for i in range(0, B, batch_size):
            feats = self.extract_features(x[i:i + batch_size])
            all_feats.append(feats.detach().cpu())
        all_feats = torch.cat(all_feats, dim=0)  # (B, feat_dim)

        mu = all_feats.mean(dim=0, keepdim=True)
        sigma = all_feats.std(dim=0, keepdim=True).clamp_min(self.eps)

        self.feat_mu.copy_(mu)
        self.feat_sigma.copy_(sigma)

    def forward(self, x: torch.Tensor, edge_index=None, graphs=None, z_prev=None) -> torch.Tensor:
        """
        Retourne des logits (B, n_classes).
        """
        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)
        else:
            z_prev = z_prev.view(x.shape[0], self.end_channels, self.n_sequences)

        if self.horizon > 0:
            x = torch.cat((x, z_prev), dim=1)

        feats = self.extract_features(x)  # (B, feat_dim)

        if self.use_feat_norm:
            feats = (feats - self.feat_mu.to(feats.device)) / self.feat_sigma.to(feats.device)

        x = self.layer_norm(feats)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        hidden = x
        logits = self.linear2(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden
