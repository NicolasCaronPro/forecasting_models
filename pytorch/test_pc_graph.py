import torch
import unittest

from forecasting_models.pytorch.pc_graph import (
    PCGraphModel, PCGraphCore, fully_connected_mask, layered_mask, bimodal_mask,
)


class TestPCGraphCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n = 12
        self.mask = fully_connected_mask(self.n)
        self.core = PCGraphCore(self.n, self.mask, init_std=0.05)
        self.clamp = torch.zeros(self.n, dtype=torch.bool)
        self.clamp[:5] = True

    def test_mask_zeroes_gradient_on_forbidden_edges(self):
        theta_masked = self.core.theta_masked()
        forbidden = self.mask == 0
        self.assertTrue(torch.all(theta_masked[forbidden] == 0))

    def test_relax_reduces_energy(self):
        x_init = torch.randn(3, self.n)
        e_before = self.core.energy(x_init).item()
        x_conv = self.core.relax(x_init, self.clamp, t_steps=25, lr_x=0.5)
        e_after = self.core.energy(x_conv).item()
        self.assertLess(e_after, e_before)

    def test_relax_respects_clamp(self):
        x_init = torch.randn(3, self.n)
        x_conv = self.core.relax(x_init, self.clamp, t_steps=10, lr_x=0.5)
        self.assertTrue(torch.allclose(x_conv[:, self.clamp], x_init[:, self.clamp]))

    def test_relax_no_grad_leaves_theta_untouched(self):
        x_init = torch.randn(3, self.n)
        x_conv = self.core.relax(x_init, self.clamp, t_steps=5, lr_x=0.5, differentiable=False)
        self.assertFalse(x_conv.requires_grad)

    def test_relax_differentiable_tracks_theta(self):
        x_init = torch.randn(3, self.n)
        x_conv = self.core.relax(x_init, self.clamp, t_steps=5, lr_x=0.5, differentiable=True)
        self.assertTrue(x_conv.requires_grad)

    def test_relax_is_batch_size_independent(self):
        """La relaxation est independante par echantillon : le resultat d'une
        ligne ne doit pas dependre du nombre de lignes traitees avec elle.

        Non-regression d'un bug reel : `relax` derivait `energy()`, qui fait
        `.mean()` sur le lot -- chaque echantillon recevait donc 1/B fois son
        propre gradient, et le pas effectif valait `lr_x / B`. Un modele
        entraine par lots de 64 puis evalue sur un lot unique de 5840 voyait
        ses noeuds libres s'ecraser d'un facteur ~12, ce qui ressemblait a un
        defaut de generalisation sans en etre un."""
        x_init = torch.randn(32, self.n)
        full = self.core.relax(x_init, self.clamp, t_steps=15, lr_x=0.5)
        by_one = torch.cat([self.core.relax(x_init[i:i + 1], self.clamp, t_steps=15, lr_x=0.5)
                            for i in range(len(x_init))])
        self.assertTrue(torch.allclose(full, by_one, atol=1e-5),
                        f'ecart max {float((full - by_one).abs().max()):.2e}')

    def test_causal_strength_matrix_shape_and_sign(self):
        m = self.core.causal_strength_matrix(n_hops=3)
        self.assertEqual(m.shape, (self.n, self.n))
        self.assertTrue(torch.all(m >= 0))

    def test_causal_strength_mediated_matches_manual_sum(self):
        start, mid, end = slice(0, 3), slice(3, 8), slice(8, self.n)
        n_hops = 3
        got = self.core.causal_strength_mediated(start, mid, end, n_hops=n_hops)

        m = self.core.theta_masked().abs()
        m1 = m[start, end]
        m2 = m[start, mid] @ m[mid, end]
        m3 = m[start, mid] @ m[mid, mid] @ m[mid, end]
        expected = m1 + m2 + m3

        self.assertEqual(got.shape, (3, self.n - 8))
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_causal_strength_mediated_ignores_non_mid_detours(self):
        # Chemin start -> start -> end (detour par un AUTRE noeud start, pas
        # un intermediaire mid) ne doit PAS etre compte, contrairement a
        # causal_strength_matrix qui, lui, l'inclurait.
        start, mid, end = slice(0, 3), slice(3, 8), slice(8, self.n)
        with torch.no_grad():
            self.core.theta.zero_()
            self.core.theta[0, 1] = 10.0   # start -> start (hors zone mid)
            self.core.theta[1, 8] = 10.0   # start -> end
            self.core.theta[0, 3] = 1.0    # start -> mid
            self.core.theta[3, 8] = 1.0    # mid -> end
        got = self.core.causal_strength_mediated(start, mid, end, n_hops=2)
        self.assertAlmostEqual(got[0, 0].item(), 1.0, places=5)  # seulement via mid, pas via node 1

    def test_strongest_paths_matches_brute_force_and_keeps_sign(self):
        start, mid, end = slice(0, 3), slice(3, 8), slice(8, self.n)
        results = self.core.strongest_paths(start, mid, end, max_hops=3)
        self.assertEqual(len(results), 3)

        theta = self.core.theta_masked()
        m_se, m_sm, m_mm, m_me = theta[start, end], theta[start, mid], theta[mid, mid], theta[mid, end]
        S, M, E = m_sm.shape[0], m_sm.shape[1], m_me.shape[1]

        val1, mids1 = results[0]
        self.assertIsNone(mids1)
        self.assertTrue(torch.allclose(val1, m_se, atol=1e-6))

        val2, mids2 = results[1]
        self.assertEqual(mids2.shape, (S, E, 1))
        for s in range(S):
            for e in range(E):
                best = max(range(M), key=lambda i: abs((m_sm[s, i] * m_me[i, e]).item()))
                self.assertEqual(mids2[s, e, 0].item(), best)
                self.assertAlmostEqual(val2[s, e].item(), (m_sm[s, best] * m_me[best, e]).item(), places=5)

        val3, mids3 = results[2]
        self.assertEqual(mids3.shape, (S, E, 2))
        s, e = 0, 0  # verifie une seule paire par force brute (couteux sinon)
        best_val, best_ij = 0.0, None
        for i in range(M):
            for j in range(M):
                v = (m_sm[s, i] * m_mm[i, j] * m_me[j, e]).item()
                if abs(v) > abs(best_val):
                    best_val, best_ij = v, (i, j)
        self.assertEqual((mids3[s, e, 0].item(), mids3[s, e, 1].item()), best_ij)
        self.assertAlmostEqual(val3[s, e].item(), best_val, places=5)

    def test_strongest_paths_rejects_max_hops_above_3(self):
        start, mid, end = slice(0, 3), slice(3, 8), slice(8, self.n)
        with self.assertRaises(NotImplementedError):
            self.core.strongest_paths(start, mid, end, max_hops=4)


class TestPCGraphModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 5
        self.in_dim = 6
        self.k_days = 2
        self.out_channels = 3
        self.common_kwargs = dict(
            in_dim=self.in_dim, k_days=self.k_days, out_channels=self.out_channels,
            n_internal=8, t_train=4, t_query=6, lr_x=0.5,
        )
        self.x = torch.randn(self.batch_size, self.in_dim, self.k_days + 1)

    def _check_forward_contract(self, model, task_type):
        model.train()
        output, logits, hidden = model(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.out_channels))
        self.assertEqual(logits.shape, (self.batch_size, self.out_channels))
        self.assertEqual(hidden.shape, (self.batch_size, model.n_internal))
        if task_type == 'classification':
            self.assertTrue(torch.allclose(output.sum(dim=-1), torch.ones(self.batch_size), atol=1e-4))
        model.eval()
        with torch.no_grad():
            output_eval, _, _ = model(self.x)
        self.assertEqual(output_eval.shape, (self.batch_size, self.out_channels))

    def test_forward_contract_classification_phantom_full(self):
        model = PCGraphModel(task_type='classification', grad_mode='phantom', topology='full', **self.common_kwargs)
        self._check_forward_contract(model, 'classification')

    def test_forward_contract_regression_bptt_layered(self):
        model = PCGraphModel(task_type='regression', grad_mode='bptt', topology='layered', **self.common_kwargs)
        self._check_forward_contract(model, 'regression')

    def test_gradient_flows_to_theta_in_phantom_mode(self):
        model = PCGraphModel(task_type='classification', grad_mode='phantom', topology='full', **self.common_kwargs)
        _, logits, _ = model(self.x)
        target = torch.randint(0, self.out_channels, (self.batch_size,))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        self.assertIsNotNone(model.core.theta.grad)
        self.assertGreater(model.core.theta.grad.abs().sum().item(), 0)

    def test_gradient_flows_to_theta_in_bptt_mode(self):
        model = PCGraphModel(task_type='classification', grad_mode='bptt', topology='full', **self.common_kwargs)
        _, logits, _ = model(self.x)
        target = torch.randint(0, self.out_channels, (self.batch_size,))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        self.assertIsNotNone(model.core.theta.grad)
        self.assertGreater(model.core.theta.grad.abs().sum().item(), 0)

    def test_wrong_input_shape_raises(self):
        model = PCGraphModel(task_type='classification', **self.common_kwargs)
        bad_x = torch.randn(self.batch_size, self.in_dim + 1, self.k_days + 1)
        with self.assertRaises(ValueError):
            model(bad_x)

    def _check_training_energy(self, model):
        model.train()
        y_clamp = torch.nn.functional.one_hot(
            torch.randint(0, self.out_channels, (self.batch_size,)), num_classes=self.out_channels
        ).float()
        energy = model.training_energy(self.x, y_clamp)
        self.assertEqual(energy.shape, (self.batch_size,))
        self.assertTrue(torch.all(energy >= 0))
        loss = energy.mean()
        loss.backward()
        self.assertIsNotNone(model.core.theta.grad)
        self.assertGreater(model.core.theta.grad.abs().sum().item(), 0)

        # Eval sous no_grad (chemin pris par Training.launch_val_test_loader) : ne doit pas planter.
        model.eval()
        with torch.no_grad():
            energy_eval = model.training_energy(self.x, y_clamp)
        self.assertEqual(energy_eval.shape, (self.batch_size,))

    def test_training_energy_phantom(self):
        model = PCGraphModel(task_type='classification', grad_mode='phantom', topology='full', **self.common_kwargs)
        self._check_training_energy(model)

    def test_training_energy_bptt(self):
        model = PCGraphModel(task_type='classification', grad_mode='bptt', topology='layered', **self.common_kwargs)
        self._check_training_energy(model)

    def test_training_energy_respects_train_clamp(self):
        # Les noeuds sensoriels ET label doivent rester fixes a leur valeur
        # clampee pendant toute la relaxation d'entrainement.
        model = PCGraphModel(task_type='classification', **self.common_kwargs)
        model.train()
        y_clamp = torch.nn.functional.one_hot(
            torch.randint(0, self.out_channels, (self.batch_size,)), num_classes=self.out_channels
        ).float()
        B = self.x.shape[0]
        x_sensory = self.x.reshape(B, -1)
        x_init = x_sensory.new_zeros(B, model.n_nodes)
        x_init[:, model.sensory_slice] = x_sensory
        x_init[:, model.label_slice] = y_clamp
        x_conv = model.core.relax(x_init, model.clamp_mask_train, model.t_train, model.lr_x)
        self.assertTrue(torch.allclose(x_conv[:, model.sensory_slice], x_sensory))
        self.assertTrue(torch.allclose(x_conv[:, model.label_slice], y_clamp))


class TestBimodalMask(unittest.TestCase):
    """`bimodal_mask` : les deux modalites ne doivent JAMAIS se voir
    directement, ni via un noeud interne partage -- uniquement via les
    labels."""

    def setUp(self):
        self.n_sensory, self.n_internal, self.out_channels = 6, 4, 2
        self.spatial = [0, 1, 2]
        self.mask = bimodal_mask(self.n_sensory, self.n_internal, self.out_channels,
                                 spatial_nodes=self.spatial, n_internal_spatial=2)
        self.int_sp = slice(6, 8)
        self.int_tp = slice(8, 10)
        self.lab = slice(10, 12)

    def test_modalities_never_share_an_internal_node(self):
        temporal = [3, 4, 5]
        self.assertEqual(self.mask[self.spatial, :][:, self.int_tp].sum().item(), 0.0)
        self.assertEqual(self.mask[temporal, :][:, self.int_sp].sum().item(), 0.0)
        # ... mais chacune est bien reliee a SON sous-bloc
        self.assertGreater(self.mask[self.spatial, :][:, self.int_sp].sum().item(), 0.0)
        self.assertGreater(self.mask[temporal, :][:, self.int_tp].sum().item(), 0.0)

    def test_keeps_layered_guarantees(self):
        sens = slice(0, 6)
        self.assertEqual(self.mask[sens, sens].sum().item(), 0.0)          # pas de sensoriel<->sensoriel
        self.assertEqual(self.mask[sens, self.lab].sum().item(), 0.0)      # pas de raccourci vers le label
        self.assertEqual(self.mask[self.lab, self.lab].sum().item(), 0.0)  # pas d'inhibition one-hot
        self.assertEqual(self.mask[6:10, 6:10].sum().item(), 0.0)          # pas d'interne<->interne
        self.assertEqual(torch.diagonal(self.mask).sum().item(), 0.0)

    def test_both_internal_blocks_reach_labels(self):
        self.assertGreater(self.mask[self.int_sp, self.lab].sum().item(), 0.0)
        self.assertGreater(self.mask[self.int_tp, self.lab].sum().item(), 0.0)

    def test_symmetric(self):
        self.assertTrue(torch.equal(self.mask, self.mask.T))

    def test_rejects_degenerate_split(self):
        for bad in (0, 4, 5):
            with self.assertRaises(ValueError):
                bimodal_mask(6, 4, 2, spatial_nodes=[0, 1], n_internal_spatial=bad)

    def test_model_expands_features_over_seq_len(self):
        """Une feature occupe `seq_len` noeuds : l'expansion feature -> noeud
        doit garder les deux modalites disjointes meme avec k_days > 0."""
        model = PCGraphModel(in_dim=4, k_days=2, out_channels=3, task_type='classification',
                             n_internal=8, topology='bimodal',
                             spatial_feature_idx=[0, 1], n_internal_spatial=3, device='cpu')
        T = model.seq_len
        mask = model.core.mask
        spatial_nodes = [f * T + t for f in (0, 1) for t in range(T)]
        temporal_nodes = [f * T + t for f in (2, 3) for t in range(T)]
        self.assertEqual(mask[spatial_nodes, :][:, model.internal_temporal_slice].sum().item(), 0.0)
        self.assertEqual(mask[temporal_nodes, :][:, model.internal_spatial_slice].sum().item(), 0.0)

    def test_forward_and_backward(self):
        model = PCGraphModel(in_dim=6, k_days=0, out_channels=3, task_type='classification',
                             n_internal=8, topology='bimodal',
                             spatial_feature_idx=[0, 1, 2], n_internal_spatial=3, device='cpu')
        x = torch.randn(5, 6, 1)
        out, logits, _ = model(x)
        self.assertEqual(out.shape, (5, 3))
        y = torch.nn.functional.one_hot(torch.randint(0, 3, (5,)), 3).float()
        model.training_energy(x.reshape(5, -1), y).mean().backward()
        grad = model.core.theta.grad
        self.assertFalse((grad[model.core.mask == 0].abs() > 0).any())

    def test_requires_spatial_indices(self):
        with self.assertRaises(ValueError):
            PCGraphModel(in_dim=6, k_days=0, out_channels=3, task_type='classification',
                         n_internal=8, topology='bimodal', device='cpu')


class TestOrdinalSoftTarget(unittest.TestCase):
    """`PCGraphTraining._ordinal_soft_target` : rendre l'energie du PC-graph
    sensible a la distance ordinale, ce que le one-hot ne fait pas."""

    @staticmethod
    def _fn():
        # Import local : garde les imports de module de ce fichier legers
        # (GNN.pytorch_model_pc_graph tire tout le pipeline). Le package
        # `GNN` se resout depuis Prediction/, qui n'est pas forcement sur le
        # sys.path du runner selon le repertoire d'ou les tests sont lances.
        import sys
        from pathlib import Path
        prediction_dir = Path(__file__).resolve().parents[3]
        if str(prediction_dir) not in sys.path:
            sys.path.insert(0, str(prediction_dir))
        from GNN.pytorch_model_pc_graph import PCGraphTraining
        # `self` n'est pas utilise sur le chemin sans `departement` (distance
        # d'indice pure) : on peut donc appeler la methode non liee avec
        # self=None. Le chemin avec departement, lui, lit la table de risque
        # et se teste sur un vrai modele, pas ici.
        return lambda *a, **kw: PCGraphTraining._ordinal_soft_target(None, *a, **kw)

    def test_peak_is_one_at_true_class(self):
        """Le maximum doit valoir 1.0 comme le one-hot : un softmax
        l'ecraserait a ~0.5 et diluerait le terme label dans l'energie."""
        y = self._fn()(torch.arange(5), 5, 1.5)
        self.assertTrue(torch.allclose(y.max(dim=1).values, torch.ones(5)))
        self.assertTrue(torch.equal(y.argmax(dim=1), torch.arange(5)))

    def test_energy_is_distance_aware(self):
        """Le point de tout l'exercice : le cout doit CROITRE avec |j - k|,
        alors qu'il est constant (2.0) avec le one-hot."""
        y = self._fn()(torch.arange(5), 5, 1.5)
        costs = [((y[k] - y[0]) ** 2).sum().item() for k in range(5)]
        for a, b in zip(costs, costs[1:]):
            self.assertLess(a, b)

        onehot = torch.eye(5)
        oh_costs = [((onehot[k] - onehot[0]) ** 2).sum().item() for k in range(1, 5)]
        self.assertEqual(len(set(oh_costs)), 1)  # le one-hot, lui, est plat

    def test_symmetric_around_true_class(self):
        y = self._fn()(torch.tensor([2]), 5, 1.5)[0]
        self.assertAlmostEqual(y[1].item(), y[3].item(), places=6)
        self.assertAlmostEqual(y[0].item(), y[4].item(), places=6)

    def test_low_temperature_approaches_one_hot(self):
        y = self._fn()(torch.tensor([2]), 5, 0.01)[0]
        self.assertTrue(torch.allclose(y, torch.tensor([0., 0., 1., 0., 0.]), atol=1e-6))

    def test_shape_and_dtype(self):
        y = self._fn()(torch.tensor([[0], [3]]), 5, 1.5)
        self.assertEqual(y.shape, (2, 5))
        self.assertEqual(y.dtype, torch.float32)


class TestCornLabelMode(unittest.TestCase):
    """`label_mode='corn'` : les noeuds de label portent [s, logits
    conditionnels] et l'ordinalite devient structurelle."""

    def _model(self, **kw):
        return PCGraphModel(in_dim=6, k_days=0, out_channels=5,
                            task_type='classification', n_internal=8,
                            topology='layered', label_mode='corn', device='cpu', **kw)

    def test_node_layout(self):
        """1 noeud scalaire + (K-1) logits = K : le nombre de noeuds ne change
        pas, donc masques et slices d'analyse restent valides."""
        m = self._model()
        self.assertEqual(m.scalar_slice.stop - m.scalar_slice.start, 1)
        self.assertEqual(m.corn_slice.stop - m.corn_slice.start, 4)
        self.assertEqual(m.label_slice.stop - m.label_slice.start, 5)
        self.assertEqual(m.corn_slice.stop, m.label_slice.stop)

    def test_probabilities_always_valid(self):
        """Le coeur de CORN : quelles que soient les valeurs des noeuds -- y
        compris aberrantes -- la lecture reste une distribution valide."""
        m = self._model()
        for logits in (torch.randn(64, 5) * 50, torch.zeros(3, 5), -torch.ones(3, 5) * 99):
            p = m.class_probs(logits)
            self.assertTrue((p >= 0).all(), 'probabilite negative')
            self.assertTrue(torch.allclose(p.sum(dim=1), torch.ones(len(p)), atol=1e-5))

    def test_cumulative_is_decreasing(self):
        """P(y>k) doit decroitre : c'est ce que le one-hot ne garantissait pas."""
        logits = torch.randn(128, 4) * 5
        F = torch.cumprod(torch.sigmoid(logits), dim=1)
        self.assertTrue((F[:, 1:] <= F[:, :-1] + 1e-6).all())

    def test_thresholds_crossed_give_expected_class(self):
        m = self._model()
        c = 3.0
        for k in range(5):
            logits = torch.full((1, 4), -c)
            logits[0, :k] = c                      # k seuils franchis -> classe k
            p = m.class_probs(torch.cat([torch.zeros(1, 1), logits], dim=1))
            self.assertEqual(int(p.argmax(dim=1)), k)

    def test_clm_tau_survives_state_dict(self):
        """`clm_tau` doit etre un BUFFER, pas un attribut Python.

        Non-regression d'un bug reel : deduit a la construction (~0.09), il
        etait perdu au rechargement et le modele repartait sur le defaut du
        constructeur (0.3). Le rapport ecart_min/tau retombait a 0.94, sous le
        ~1.5 requis, et les classes intermediaires se vidaient -- alors que les
        donnees franchissaient bien les seuils."""
        m = self._model()
        m.set_clm_thresholds([0.02, 0.13, 0.22, 0.33])
        m.set_clm_tau(0.037)
        self.assertIn('clm_tau', m.state_dict())

        reloaded = self._model()
        self.assertNotAlmostEqual(float(reloaded.clm_tau), 0.037, places=4)
        reloaded.load_state_dict(m.state_dict())
        self.assertAlmostEqual(float(reloaded.clm_tau), 0.037, places=6)
        self.assertTrue(torch.allclose(reloaded.clm_thresholds, m.clm_thresholds, atol=1e-6))

    def test_per_department_thresholds(self):
        """Un jeu de seuils par departement : le MEME `s` doit pouvoir tomber
        dans des classes differentes selon la zone."""
        m = PCGraphModel(in_dim=6, k_days=0, out_channels=5, task_type='classification',
                         n_internal=8, topology='layered', label_mode='clm',
                         clm_tau=0.05, n_departements=3, device='cpu')
        m.set_clm_dept_ids([1, 6, 25])
        m.set_clm_thresholds(torch.tensor([[0.00, 0.30, 0.50, 0.70],
                                           [-0.20, 0.20, 0.45, 0.70],
                                           [0.20, 0.40, 0.55, 0.70]]))
        self.assertEqual(m.clm_thresholds.shape, (3, 4))

        s = torch.full((3, 1), 0.10)                 # meme score
        m.set_current_departement(torch.tensor([1, 6, 25]))
        pred = m.class_probs(s).argmax(dim=1)
        self.assertEqual(int(pred[1]), 1)            # dept 6 : au-dessus de -0.20
        self.assertEqual(int(pred[2]), 0)            # dept 25 : en dessous de 0.20
        self.assertNotEqual(int(pred[1]), int(pred[2]))

    def test_department_ids_survive_state_dict(self):
        m = PCGraphModel(in_dim=6, k_days=0, out_channels=5, task_type='classification',
                         n_internal=8, topology='layered', label_mode='clm',
                         n_departements=3, device='cpu')
        m.set_clm_dept_ids([1, 6, 25])
        self.assertIn('clm_dept_ids', m.state_dict())
        self.assertNotIn('_current_dept', m.state_dict())   # transitoire, jamais persiste

    def test_rejects_unknown_label_mode(self):
        with self.assertRaises(ValueError):
            PCGraphModel(in_dim=6, k_days=0, out_channels=5, task_type='classification',
                         n_internal=8, label_mode='nimportequoi', device='cpu')

    def test_forward_and_backward(self):
        m = self._model()
        x = torch.randn(5, 6, 1)
        out, logits, _ = m(x)
        self.assertEqual(out.shape, (5, 5))
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(5), atol=1e-5))
        y = torch.randn(5, 5)
        m.training_energy(x.reshape(5, -1), y).mean().backward()
        self.assertFalse((m.core.theta.grad[m.core.mask == 0].abs() > 0).any())


if __name__ == '__main__':
    unittest.main()
