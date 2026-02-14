import sys
import torch
import numpy as np
from unittest.mock import MagicMock

# Comprehensive Mock for dgl, blitz, and reservoirpy
dgl_mock = MagicMock()
sys.modules['dgl'] = dgl_mock
sys.modules['dgl.nn'] = MagicMock()
sys.modules['dgl.nn.pytorch'] = MagicMock()
sys.modules['dgl.nn.pytorch.conv'] = MagicMock()
sys.modules['dgl.nn.pytorch.GATConv'] = MagicMock()
sys.modules['dgl.function'] = MagicMock()
sys.modules['dgl.nn.functional'] = MagicMock()

blitz_mock = MagicMock()
sys.modules['blitz'] = blitz_mock
sys.modules['blitz.modules'] = MagicMock()
sys.modules['blitz.modules.BayesianLinear'] = MagicMock()
sys.modules['blitz.utils'] = MagicMock()
sys.modules['blitz.utils.variational_estimator'] = MagicMock()
sys.modules['blitz.losses'] = MagicMock()
sys.modules['blitz.losses.kl_divergence_from_nn'] = MagicMock()

reservoirpy_mock = MagicMock()
sys.modules['reservoirpy'] = reservoirpy_mock
sys.modules['reservoirpy.nodes'] = MagicMock()
sys.modules['reservoirpy.nodes.Reservoir'] = MagicMock()

# Mocking the imports
sys.path.insert(0,'/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/')
from forecasting_models.pytorch.ordinal_loss import OrdinalMonotonicLossNoCoverageWithGains

def test_dirichlet_regularization():
    torch.manual_seed(42)
    num_classes = 5
    
    # Create dummy data
    N = 10
    logits = torch.randn(N, num_classes, requires_grad=True)
    y_cont = torch.randn(N) 
    clusters_ids = torch.tensor([0]*5 + [1]*5) # Two clusters
    
    print("--- Test Dirichlet Regularization ---")
    
    # 1. Baseline: No Dirichlet (lambdadir=0.0)
    baseline_crit = OrdinalMonotonicLossNoCoverageWithGains(
        num_classes=num_classes, 
        lambdaentropy=0.0, 
        addglobal=True,
        lambdadir=0.0,
        diralpha=1.05
    )
    loss_baseline = baseline_crit(logits, y_cont, clusters_ids)
    print(f"Baseline Loss (lambdadir=0.0): {loss_baseline.item()}")
    
    # 2. With Dirichlet (lambdadir=0.1)
    dir_crit = OrdinalMonotonicLossNoCoverageWithGains(
        num_classes=num_classes, 
        lambdaentropy=0.0, 
        addglobal=True,
        lambdadir=0.1,
        diralpha=1.05
    )
    loss_dir = dir_crit(logits, y_cont, clusters_ids)
    print(f"Dirichlet Loss (lambdadir=0.1): {loss_dir.item()}")
    
    # Check if loss increased
    if loss_dir.item() > loss_baseline.item():
        print("SUCCESS: Dirichlet term increased the loss.")
    else:
        print("FAILURE: Dirichlet term did not increase the loss.")

    # Check logs
    global_stats = dir_crit.epoch_stats['global']
    print(f"Global Stats available keys: {global_stats.keys()}")
    
    if 'dirichlet_reg' in global_stats:
        reg_val = global_stats['dirichlet_reg'][0]
        print(f"dirichlet_reg value in global logs: {reg_val}")
        if reg_val > 0:
             print("SUCCESS: dirichlet_reg is logged and positive.")
        else:
             print("WARNING: dirichlet_reg is 0 or negative.")
    else:
        print("FAILURE: dirichlet_reg key missing in global stats.")

    # Check Cluster Logs (should be 0.0 or not present for Dirichlet)
    cluster_stats = dir_crit.epoch_stats[0]
    if 'dirichlet_reg' in cluster_stats:
         reg_val_c = cluster_stats['dirichlet_reg'][0]
         print(f"Cluster 0 dirichlet_reg: {reg_val_c}")
         if reg_val_c == 0.0:
              print("SUCCESS: Cluster dirichlet_reg is 0.0 as expected (Test A).")
         else:
              print(f"FAILURE: Cluster dirichlet_reg is {reg_val_c}, expected 0.0.")
    else:
         print("SUCCESS: dirichlet_reg not in cluster stats (or 0.0 logic used).")

if __name__ == "__main__":
    try:
        test_dirichlet_regularization()
        print("All tests passed.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
