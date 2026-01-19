import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEMSE(nn.Module):
    def __init__(self, lambda_dist=1.0, mask_on_boundary=True):
        super(BCEMSE, self).__init__()
        self.lambda_dist = lambda_dist
        self.mask_on_boundary = mask_on_boundary
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, target):
        """
        logits: (B, H, W, 2)
        target: (B, H, W, 2)
        """
        # Ensure logits are (B, H, W, 2)
        if logits.dim() == 4 and logits.shape[1] == 2 and logits.shape[-1] != 2:
             logits = logits.permute(0, 2, 3, 1)
        
        # Ensure target is (B, H, W, 2)
        if target.dim() == 4 and target.shape[1] == 2 and target.shape[-1] != 2:
             target = target.permute(0, 2, 3, 1)

        boundary_logits = logits[..., 0]
        dist_pred = logits[..., 1]
        
        boundary_target = target[..., 0]
        dist_target = target[..., 1]
        
        loss_boundary = self.bce(boundary_logits, boundary_target.float())
        
        loss_dist = self.mse(dist_pred, dist_target)
        
        if self.mask_on_boundary:
            # Mask distance loss where boundary_target is 1
            # Assuming boundary_target is 0 or 1
            mask = (1 - boundary_target).float()
            loss_dist = loss_dist * mask
            
        loss = loss_boundary + self.lambda_dist * loss_dist.mean()
        
        return loss

class WCEMSE(nn.Module):
    """
    Weighted CE + MSE for vector predictions.

    logits: (B, 3) -> [bg_logit, edge_logit, dist_pred]
    target: (B, 2) -> [class_idx (0/1), dist_target]
    sample_weight: optional (B,) or scalar
    """
    def __init__(self, lambda_dist=1.0, class_weights=None, num_classes=5):
        super().__init__()
        self.lambda_dist = lambda_dist
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    def _ensure_weight_b(self, sample_weight: torch.Tensor, B: int, device):
        if sample_weight is None:
            return None
        if not torch.is_tensor(sample_weight):
            sample_weight = torch.tensor(sample_weight, device=device)
        if sample_weight.dim() == 0:
            sample_weight = sample_weight.expand(B)
        if sample_weight.dim() != 1 or sample_weight.shape[0] != B:
            raise ValueError(f"sample_weight must be (B,) or scalar, got {tuple(sample_weight.shape)}")
        return sample_weight.float()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, sample_weight: None):
        if logits.dim() != 2 or logits.shape[1] != 3:
            raise ValueError(f"logits must be (B,3), got {tuple(logits.shape)}")
        if target.dim() != 2 or target.shape[1] != 2:
            raise ValueError(f"target must be (B,2), got {tuple(target.shape)}")
        
        B = logits.shape[0]
        device = logits.device

        # Split predictions
        class_logits = logits[:, 0:2]          # (B,2) : bg vs edge
        dist_pred    = logits[:, 2]            # (B,)

        # Split targets
        class_target = target[:, 0].long()     # (B,) values in {0,1}
        dist_target  = target[:, 1].float()    # (B,)

        w = self._ensure_weight_b(sample_weight, B, device)  # (B,) or None

        # --- CE (per-sample) ---
        loss_ce = self.ce(class_logits, class_target)        # (B,)

        # --- MSE (per-sample), only for non-edge samples ---
        loss_mse = self.mse(dist_pred, dist_target)          # (B,)
        reg_mask = (class_target == 0).float()               # (B,) 1=non-frontière
        loss_mse = loss_mse * reg_mask

        if w is not None:
            # Weighted mean CE
            denom_ce = w.sum().clamp_min(1e-8)
            loss_ce_mean = (loss_ce * w).sum() / denom_ce

            # Weighted mean MSE on valid reg samples only
            denom_mse = (w * reg_mask).sum().clamp_min(1e-8)
            loss_mse_mean = (loss_mse * w).sum() / denom_mse
        else:
            loss_ce_mean = loss_ce.mean()

            denom_mse = reg_mask.sum().clamp_min(1e-8)
            loss_mse_mean = loss_mse.sum() / denom_mse

        return loss_ce_mean + self.lambda_dist * loss_mse_mean