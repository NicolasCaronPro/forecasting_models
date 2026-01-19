import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from forecasting_models.pytorch.student_distillation import StudentMLP
from forecasting_models.pytorch.distillation_utils import Adapter, FitNet, multi_teacher_kd_loss, lht_loss, angle_triplet_loss

class TestAdatativeMLP(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 4
        self.input_dim = 128
        self.num_classes = 10
        self.num_teachers = 2
        self.teacher_dim = 128
        self.student_rep_dim = 128
        self.n_group = 3
        
        # Student (MLP)
        self.student = StudentMLP(
            n_group=self.n_group,
            input_dim=self.input_dim,
            mlp_hidden_dim=128,
            embedding_dim=self.student_rep_dim,
            num_classes=self.num_classes,
            device=self.device,
            task_type='classification'
        ).to(self.device)
        
        # Adapter
        self.adapter = Adapter(d=self.student_rep_dim, num_teachers=self.num_teachers).to(self.device)
        
        # FitNets
        self.fitnets = nn.ModuleList([
            FitNet(c_student=self.student_rep_dim, c_teacher=self.teacher_dim).to(self.device)
            for _ in range(self.num_teachers)
        ])
        
    def test_forward_and_loss(self):
        # Dummy inputs (B, input_dim)
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        # Student Forward
        # Returns: output, final_logits, hidden_features
        output, final_logits, hidden_features = self.student(x)
        
        # Check stored intermediates
        self.assertEqual(len(self.student.logits_list), self.n_group)
        self.assertEqual(len(self.student.hidden_features), self.n_group)
        
        self.assertEqual(len(hidden_features), self.n_group)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertEqual(final_logits.shape, (self.batch_size, self.num_classes))
        
        student_logits = final_logits
        student_rep = hidden_features[-1] # (B, embedding_dim)
        
        # Dummy Teachers
        t_logits_list = [
            torch.randn(self.batch_size, self.num_classes).to(self.device) for _ in range(self.num_teachers)
        ]
        # Teacher features: (B, teacher_dim) or (B, C, H, W)
        # Let's test with 4D features to verify flattening in FitNet/LHT
        t_feat_list = [
            torch.randn(self.batch_size, 8, 4, 4).to(self.device) for _ in range(self.num_teachers)
        ]
        # 8*4*4 = 128 = teacher_dim
        
        # 1. CE Loss
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # 2. Adapter
        weights = self.adapter(student_rep)
        self.assertEqual(weights.shape, (self.batch_size, self.num_teachers))
        
        # 3. LKD Loss
        T = 5.0
        lkd, fused_soft, t_soft_all = multi_teacher_kd_loss(
            student_logits, t_logits_list, weights, T=T
        )
        
        # 4. LAngle Loss
        s_soft = F.softmax(student_logits / T, dim=-1)
        l_angle = angle_triplet_loss(fused_soft, s_soft)
        
        # 5. LHT Loss
        # Map all teachers to last student group
        student_group_feats = [hidden_features[-1] for _ in range(self.num_teachers)]
        l_ht = lht_loss(t_feat_list, student_group_feats, self.fitnets)
        
        # Total Loss
        loss = ce_loss + lkd + l_angle + l_ht
        
        # Backward
        loss.backward()
        
        print("Test passed: Forward and Backward successful")
        print(f"Loss: {loss.item()}")

if __name__ == '__main__':
    unittest.main()
