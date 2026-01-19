import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentMLPConcat(nn.Module):
    """
    Student MLP avec n_group blocs séquentiels.
    Chaque bloc produit une embedding, et on concatène toutes les embeddings
    pour calculer une représentation finale (end_rep) de dimension end_channels,
    puis les logits.

    hidden_features[i] pourra être aligné avec le groupe de teachers i,
    et hidden_features[-1] = représentation finale après end_layer.
    """
    def __init__(
        self,
        n_group=3,
        input_dim=128,
        mlp_hidden_dim=128,
        embedding_dim=128,
        end_channels=64,
        out_channels=2,
        task_type='classification'
    ):
        super().__init__()
        self.n_group = n_group
        self.input_dim = input_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embedding_dim = embedding_dim
        self.end_channels = end_channels
        self.out_channels = out_channels
        self.task_type = task_type

        # Blocs MLP séquentiels
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for _ in range(n_group):
            block = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, embedding_dim),
            )
            self.blocks.append(block)
            # Le bloc suivant prend en entrée l'embedding précédente
            in_dim = embedding_dim

        # Couche "end" : prend la concaténation des n_group embeddings
        self.end_layer = nn.Linear(embedding_dim * n_group, end_channels)

        # Projection finale vers les logits
        self.output_layer = nn.Linear(end_channels, out_channels)

    def forward(self, x, z_prev=None):
        # Si x est temporel (B, D, T), on prend le dernier horizon
        if x.dim() == 3:
            x = x[:, :, -1]  # -> (B, D)

        hidden_features = []
        h = x
        for block in self.blocks:
            h = block(h)               # (B, embedding_dim)
            hidden_features.append(h)

        # Concaténation des embeddings des n groupes
        concat = torch.cat(hidden_features, dim=1)        # (B, embedding_dim * n_group)

        # Passage par la couche end + ReLU
        end_rep = F.relu(self.end_layer(concat))          # (B, end_channels)

        # On ajoute aussi cette représentation finale dans hidden_features
        hidden_features.append(end_rep)

        # Logits calculés sur cette représentation finale
        final_logits = self.output_layer(end_rep)         # (B, out_channels)

        if self.task_type == 'classification':
            output = F.softmax(final_logits, dim=1)
        else:
            output = final_logits

        # hidden_features = [feat_block1, feat_block2, ..., feat_blockN, end_rep]
        return output, final_logits, hidden_features

class StudentMLP(nn.Module):
    """
    StudentMLP model for AdaptiveMLP distillation (Sequential Version).
    
    Architecture:
    - n_group blocks, connected sequentially.
    - Input to block i is output of block i-1.
    - Last block -> end_layer -> logits.
    - hidden_features[i] = sortie block i
      hidden_features[-1] = représentation finale après end_layer
    """
    def __init__(
        self,
        n_group=3,
        input_dim=128,
        mlp_hidden_dim=128,
        embedding_dim=128,
        out_channels=2,
        end_channels=64,
        device='cpu',
        task_type='classification'
    ):
        super(StudentMLP, self).__init__()
        self.n_group = n_group
        self.input_dim = input_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.end_channels = end_channels
        self.task_type = task_type

        # Create n_group blocks (séquentiels)
        self.blocks = nn.ModuleList()
        current_dim = input_dim
        for _ in range(n_group):
            block = nn.Sequential(
                nn.Linear(current_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_hidden_dim, embedding_dim),
            )
            self.blocks.append(block)
            current_dim = embedding_dim
        
        # Couche "end" : projette la dernière embedding vers end_channels
        self.end_layer = nn.Linear(embedding_dim, end_channels)

        # Projection finale F : logits à partir de end_channels
        self.output_layer = nn.Linear(end_channels, out_channels)

        # (facultatif) stockage pour accès externe
        self.logits_list = []
        self.hidden_features = []

    def forward(self, x, z_prev=None):
        # z_prev est là pour compatibilité avec ton Training.launch_batch

        # Si x est temporel (B, D, T), on prend le dernier horizon
        if x.dim() == 3 and x.shape[-1] > 1:
            x = x[:, :, -1]

        hidden_features = []
        current_input = x
        
        # Passage séquentiel dans les blocs
        for block in self.blocks:
            out = block(current_input)   # (B, embedding_dim)
            hidden_features.append(out)
            current_input = out
        
        # Représentation de la dernière embedding via end_layer + ReLU
        last_feat = hidden_features[-1]               # (B, embedding_dim)
        end_rep = F.relu(self.end_layer(last_feat))   # (B, end_channels)

        # On ajoute aussi cette représentation finale dans hidden_features
        hidden_features.append(end_rep)
        
        # Logits calculés sur cette représentation finale
        final_logits = self.output_layer(end_rep)     # (B, out_channels)
        
        if self.task_type == 'classification':
            output = F.softmax(final_logits, dim=1)
        else:
            output = final_logits
            
        # Stockage si tu veux y accéder dans model_distillation_loss
        self.hidden_features = hidden_features
        
        return output, final_logits, hidden_features