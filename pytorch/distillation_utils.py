"""
Distillation utilities for knowledge distillation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationMLP(nn.Module):
    """
    MLP that learns to combine multiple teacher logits into an ensemble embedding.
    """
    def __init__(self, num_teachers, num_classes, embedding_dim, mlp_hidden_dim=128):
        super().__init__()
        self.num_teachers = num_teachers
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        input_dim = num_teachers * embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
    
    def forward(self, teacher_logits):
        if teacher_logits.ndim == 3:
            B, M, C = teacher_logits.shape
            assert M == self.num_teachers
        elif teacher_logits.ndim == 2:
            B, C = teacher_logits.shape
        elif teacher_logits.ndim == 4:
            B, M, _, C = teacher_logits.shape
            teacher_logits = teacher_logits[:, :, 0, :]
        else:
            raise ValueError(f'teacher_logits shape should (B, M, C) or (B, C) got {teacher_logits.shape}')
        
        assert C == self.num_classes
        concat_logits = teacher_logits.reshape(B, M * C)
        ensemble_embedding = self.mlp(concat_logits)
        return ensemble_embedding

class RelationAttention(nn.Module):
    """
    Attention-based module that learns to combine multiple teacher logits into an ensemble embedding.
    """
    def __init__(self, num_teachers, num_classes, embedding_dim, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_teachers = num_teachers
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        self.output_proj = nn.Sequential(
            nn.Linear(num_teachers * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, teacher_logits):
        if teacher_logits.ndim == 3:
            B, M, C = teacher_logits.shape
            assert M == self.num_teachers
        elif teacher_logits.ndim == 2:
            B, C = teacher_logits.shape
            teacher_logits = teacher_logits.unsqueeze(0)
            M = 1
        elif teacher_logits.ndim == 4:
            B, M, _, C = teacher_logits.shape
            teacher_logits = teacher_logits[:, :, 0, :]
        else:
            raise ValueError(f'teacher_logits shape should (B, M, C) or (B, C) got {teacher_logits.shape}')
        
        assert C == self.embedding_dim
        x = teacher_logits.permute(1, 0, 2)
        x = self.input_proj(x)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output
        x = x.permute(1, 0, 2).reshape(B, M * self.hidden_dim)
        ensemble_embedding = self.output_proj(x)
        return ensemble_embedding

class Adapter(nn.Module):
    """
    Implémente l'importance des teachers w_{t,i} à partir de la rep du student.
    student_rep: (B, d)
    """
    def __init__(self, d, num_teachers):
        super().__init__()
        self.num_teachers = num_teachers
        self.teacher_factors = nn.Parameter(
            torch.randn(num_teachers, d) * 0.01
        )  # h_t
        self.m_vec = nn.Parameter(
            torch.randn(d) * 0.01
        )  # vecteur global m

    def forward(self, student_rep):
        # student_rep: (B, d)
        B, d = student_rep.shape
        ht = self.teacher_factors.unsqueeze(0)  # (1, m, d)
        s = student_rep.unsqueeze(1)           # (B, 1, d)
        prod = ht * s                          # (B, m, d)
        scores = torch.einsum('bmd,d->bm', prod, self.m_vec)  # (B, m)
        weights = torch.softmax(scores, dim=1)  # (B, m)
        return weights

class FitNet(nn.Module):
    """
    Projection des features du student vers l'espace du teacher.
    Adapted for MLP student (2D features).
    """
    def __init__(self, c_student, c_teacher):
        super().__init__()
        self.proj = nn.Linear(c_student, c_teacher)

    def forward(self, v):
        # v: (B, c_student)
        if v.ndim > 2:
            v = v.view(v.size(0), -1)
        return self.proj(v)


def multi_teacher_kd_loss(student_logits,
                          teacher_logits_list,
                          weights,
                          T=5.0):
    """
    LKD avec soft-targets fusionnées (Eq. (7) + KD).
    student_logits: (B, C)
    teacher_logits_list: liste de m tensors (B, C)
    weights: (B, m)
    """
    B, C = student_logits.shape
    m = len(teacher_logits_list)

    t_logits = torch.stack(teacher_logits_list, dim=1)      # (B, m, C)
    t_soft_all = torch.softmax(t_logits / T, dim=-1)        # (B, m, C)

    w = weights.unsqueeze(-1)                               # (B, m, 1)
    fused_soft = (w * t_soft_all).sum(dim=1)                # (B, C)

    log_s = torch.log_softmax(student_logits / T, dim=-1)   # (B, C)

    kd = F.kl_div(log_s, fused_soft, reduction='batchmean') * (T * T)
    return kd, fused_soft, t_soft_all  # on renvoie aussi les softs pour LAngle

def multi_teacher_kd_loss_global_weights(student_logits,
                                         teacher_logits_list,
                                         weights=None,
                                         T=5.0):
    """
    LKD avec soft-targets fusionnées (Eq. (7) + KD), 
    mais ici `weights` est un poids GLOBAL par teacher, 
    identique pour tous les exemples du batch.

    Args:
        student_logits: (B, C)
        teacher_logits_list: liste de m tensors (B, C)
        weights: (m,) ou None
                 - si None -> poids uniformes 1/m
        T: température

    Returns:
        kd:      scalaire, loss de distillation
        fused_soft: (B, C), soft-target fusionnée
        t_soft_all: (B, m, C), soft-targets de chaque teacher
    """
    device = student_logits.device
    B, C = student_logits.shape
    m = len(teacher_logits_list)
    assert m > 0, "teacher_logits_list ne doit pas être vide"

    # Stack des logits teachers : (B, m, C)
    t_logits = torch.stack(teacher_logits_list, dim=1)          # (B, m, C)
    t_soft_all = torch.softmax(t_logits / T, dim=-1)            # (B, m, C)

    # Gestion des poids globaux par teacher
    if weights is None:
        # Poids uniformes
        w = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)
    else:
        w = torch.as_tensor(weights, device=device, dtype=torch.float32)
        assert w.shape[0] == m, "weights doit être de taille (m,)"
        # Normalisation de sécurité
        w_sum = w.sum()
        if w_sum.abs() < 1e-8:
            w = torch.full_like(w, 1.0 / m)
        else:
            w = w / w_sum

    # Mise en forme pour broadcast : (1, m, 1)
    w_broadcast = w.view(1, m, 1)

    # Soft-target fusionnée : somme_t w_t * p_t(x)
    fused_soft = (w_broadcast * t_soft_all).sum(dim=1)          # (B, C)

    # Log-softmax du student
    log_s = torch.log_softmax(student_logits / T, dim=-1)       # (B, C)

    # KL distillation (student || fused_teacher)
    kd = F.kl_div(log_s, fused_soft, reduction='batchmean') * (T * T)

    return kd, fused_soft, t_soft_all

def lht_loss(teacher_feats, student_feats, fitnets):
    """
    LHT = somme des MSE sur les features intermédiaires.
    teacher_feats: liste m de features teacher (B, ...)
    student_feats: liste m de features student (B, C_s)
    fitnets:      liste m de modules FitNet
    """
    loss = 0.0
    for ut, vl, ft in zip(teacher_feats, student_feats, fitnets):
        proj_student = ft(vl)
        
        # Flatten teacher features if necessary to match projected student
        if ut.ndim > 2:
            ut = ut.view(ut.size(0), -1)
            
        loss = loss + F.mse_loss(proj_student, ut)
    return loss


def angle_triplet_loss(teacher_soft, student_soft, num_triplets=500, delta=1.0):
    """
    LAngle sur des triplets (i,j,k) échantillonnés du batch.
    teacher_soft : (B, C)  = soft-targets intégrés (y_T)
    student_soft : (B, C)  = soft-targets student (y_S)
    """
    device = teacher_soft.device
    B = teacher_soft.size(0)
    if B < 3:
        return torch.tensor(0.0, device=device)

    num_triplets = min(num_triplets, B * (B - 1) * (B - 2))

    idx_i = torch.randint(0, B, (num_triplets,), device=device)
    idx_j = torch.randint(0, B, (num_triplets,), device=device)
    idx_k = torch.randint(0, B, (num_triplets,), device=device)

    mask_same = (idx_i == idx_j) | (idx_k == idx_j)
    while mask_same.any():
        idx_j[mask_same] = torch.randint(0, B, (mask_same.sum(),), device=device)
        mask_same = (idx_i == idx_j) | (idx_k == idx_j)

    t_i, t_j, t_k = teacher_soft[idx_i], teacher_soft[idx_j], teacher_soft[idx_k]
    s_i, s_j, s_k = student_soft[idx_i], student_soft[idx_j], student_soft[idx_k]

    t_ij = t_i - t_j
    t_kj = t_k - t_j
    s_ij = s_i - s_j
    s_kj = s_k - s_j

    def normalize(v, eps=1e-8):
        return v / (v.norm(dim=-1, keepdim=True) + eps)

    t_ij_n = normalize(t_ij)
    t_kj_n = normalize(t_kj)
    s_ij_n = normalize(s_ij)
    s_kj_n = normalize(s_kj)

    t_angle = (t_ij_n * t_kj_n).sum(dim=-1).clamp(-1.0, 1.0)
    s_angle = (s_ij_n * s_kj_n).sum(dim=-1).clamp(-1.0, 1.0)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean', beta=delta)
    return loss


def confidence_distillation_loss(student_logits,
                                 student_feat,
                                 teacher_logits_list,
                                 teacher_feat_list,
                                 labels,
                                 fitnets,
                                 teacher_classifiers,
                                 alpha=1.0,
                                 beta=50.0,
                                 T=4.0):
    """
    Implémentation de CA-MKD (Confidence-Aware Multi-Teacher KD)
    
    L = alpha * L_KD + beta * L_inter
    
    (L_CE est supposée être calculée ailleurs)

    - L_KD     : logits distillation avec poids w_KD (confiance prédictions finales)
    - L_inter  : feature distillation avec poids w_inter (confiance dans l'espace des features)
    """
    device = student_logits.device
    B, C = student_logits.shape
    K = len(teacher_logits_list)

    # ------------------------------------------------------------
    # 1) L_KD : logits distillation avec poids w_KD
    # ------------------------------------------------------------

    # One-hot des labels pour calcul CE manuelle (pour les poids)
    y_onehot = F.one_hot(labels, num_classes=C).float()  # (B, C)

    # Stack des logits teachers: (B, K, C)
    t_logits = torch.stack(teacher_logits_list, dim=1)  # (B, K, C)
    # Softmax sans température pour la CE de confiance
    t_probs = torch.softmax(t_logits, dim=-1)           # (B, K, C)

    # L_CE^KD,k pour chaque sample et chaque teacher
    # shape: (B, K)
    # - somme_c y_c log(p_k,c)
    teacher_ce_kd = -(y_onehot.unsqueeze(1) * torch.log(t_probs + 1e-12)).sum(dim=-1)

    # On calcule w_KD,k comme Eq. (2), mais ici de façon sample-wise :
    # w_k = 1/(K-1) * ( 1 - exp(L_k)/sum_j exp(L_j) )
    exp_ce = torch.exp(teacher_ce_kd)                      # (B, K)
    denom = exp_ce.sum(dim=1, keepdim=True) + 1e-12        # (B, 1)
    frac = exp_ce / denom                                  # (B, K)
    w_kd = (1.0 / (K - 1)) * (1.0 - frac)                  # (B, K)

    # Logits distillation avec température T (Hinton-style)
    s_log_soft = torch.log_softmax(student_logits / T, dim=-1)  # (B, C)
    t_soft_all = torch.softmax(t_logits / T, dim=-1)            # (B, K, C)

    # Fusion des teachers: moyenne pondérée par w_KD
    # w_kd: (B, K) -> (B, K, 1)
    w_kd_exp = w_kd.unsqueeze(-1)
    fused_soft = (w_kd_exp * t_soft_all).sum(dim=1)             # (B, C)

    # KL(student || fused_teacher)
    kd_loss = F.kl_div(s_log_soft, fused_soft, reduction='batchmean') * (T * T)

    # ------------------------------------------------------------
    # 2) L_inter : feature distillation avec poids w_inter
    # ------------------------------------------------------------
    
    # Calcul des poids w_inter
    teacher_ce_inter = []
    
    # On itère sur les teachers pour calculer leur confiance intermédiaire
    for k in range(K):
        # Projection student -> teacher k space
        student_feat_proj = fitnets[k](student_feat) 
        
        # teacher_classifiers[k]: (B, D_t) -> (B, C)
        z_s2t = teacher_classifiers[k](student_feat_proj) # (B, C)
        
        p_s2t = torch.softmax(z_s2t, dim=-1)            # (B, C)
        ce_k = -(y_onehot * torch.log(p_s2t + 1e-12)).sum(dim=-1)   # (B,)
        teacher_ce_inter.append(ce_k)

    # (B, K)
    teacher_ce_inter = torch.stack(teacher_ce_inter, dim=1)

    # w_inter,k comme Eq. (6), sample-wise :
    exp_ce_inter = torch.exp(teacher_ce_inter)
    denom_inter = exp_ce_inter.sum(dim=1, keepdim=True) + 1e-12
    frac_inter = exp_ce_inter / denom_inter
    w_inter = (1.0 / (K - 1)) * (1.0 - frac_inter)              # (B, K)

    # MSE pondérée entre teacher features et student feature projetée
    l_inter_per_sample = 0.0
    for k in range(K):
        t_feat_k = teacher_feat_list[k] # (B, D_t)
        if t_feat_k.ndim > 2:
             t_feat_k = t_feat_k.view(t_feat_k.size(0), -1)
             
        s_feat_proj_k = fitnets[k](student_feat) # (B, D_t)
        
        mse_k = ((t_feat_k - s_feat_proj_k) ** 2).mean(dim=-1) # (B,)
        l_inter_per_sample += w_inter[:, k] * mse_k

    inter_loss = l_inter_per_sample.mean()

    # ------------------------------------------------------------
    # 3) Loss totale (sans CE)
    # ------------------------------------------------------------
    total_loss = alpha * kd_loss + beta * inter_loss

    return total_loss
