"""PC-graph : predictive coding sur topologie de graphe arbitraire.

Implementation adaptee de Salvatori, Pinchetti, Millidge, Song, Bao, Bogacz,
Lukasiewicz, "Learning on Arbitrary Graph Topologies via Predictive Coding",
NeurIPS 2022 (arXiv:2201.13180) -- prototype explore dans
Prediction/GNN/PC-graph/pc_graphs_mnist.ipynb.

Adaptation au pipeline du projet (voir pytorch_model_pc_graph.py) : l'article
clampe sensoriel+label pendant l'entrainement et fait une seule mise a jour
des poids a l'equilibre (Algorithme 1 litteral). Le pipeline calcule toujours
sa loss *en dehors* du modele (Training.calculate_loss sur les `logits`
retournes par `forward`), donc le label ne peut pas etre clampe : PCGraphModel
clampe uniquement les noeuds sensoriels, relaxe, et expose la valeur
d'equilibre des noeuds de label comme `logits` -- une couche d'equilibre
differentiable (esprit Deep Equilibrium Model), entrainee par la loss/backward
standard du reste du pipeline.
"""
import torch
import torch.nn as nn


def fully_connected_mask(n: int) -> torch.Tensor:
    return torch.ones(n, n) - torch.eye(n)


def layered_mask(group_sizes: list) -> torch.Tensor:
    """Connexions autorisees seulement entre groupes adjacents (sensoriel <->
    interne <-> label). Force l'information a passer par les noeuds internes
    plutot que d'utiliser un raccourci direct sensoriel -> label."""
    n = sum(group_sizes)
    group_id = torch.zeros(n, dtype=torch.long)
    start = 0
    for k, size in enumerate(group_sizes):
        group_id[start:start + size] = k
        start += size
    diff = (group_id.view(-1, 1) - group_id.view(1, -1)).abs()
    return (diff == 1).float()


def bimodal_mask(n_sensory: int, n_internal: int, out_channels: int,
                 spatial_nodes, n_internal_spatial: int) -> torch.Tensor:
    """Variante de `layered_mask` qui SEPARE les deux modalites du jeu de
    donnees : le bloc interne est coupe en deux sous-blocs disjoints, l'un
    dedie aux features spatiales, l'autre aux temporelles. Chaque sous-bloc
    ne voit que sa propre modalite ; ils ne se rejoignent qu'au niveau des
    labels.

    Motivation (verifiee empiriquement sur ce jeu de donnees) : les features
    spatiales sont STRICTEMENT constantes a l'interieur d'un cluster (100%
    de leur variance est inter-cluster, contre ~5% pour les temporelles).
    Avec un bloc interne unique et partage, elles n'apportent donc que
    l'identite du cluster -- soit 16 lignes d'information distinctes -- et
    monopolisent malgre tout une part du classement causal en encodant le
    taux de base de sinistres de chaque cluster. Les separer force chaque
    sous-bloc a expliquer sa propre modalite avant toute combinaison.

    Conserve les proprietes de `layered_mask` : aucune arete
    sensoriel<->sensoriel, sensoriel->label directe, label<->label ni
    interne<->interne. La disposition des noeuds (sensoriel, puis interne,
    puis label) est inchangee, donc `sensory_slice` / `internal_slice` /
    `label_slice` gardent le meme sens et tout l'outillage d'analyse
    fonctionne sans modification.

    `spatial_nodes` : indices (dans le bloc sensoriel) des noeuds spatiaux ;
    tous les autres sont traites comme temporels. Un noeud liste a la fois
    dans les deux n'a pas de sens ici -- le complement est automatique.
    `n_internal_spatial` : taille du sous-bloc interne spatial (le reste va
    au temporel). A dimensionner petit : mettre autant d'internes que pour
    le temporel garantirait la memorisation des 16 profils de cluster.
    """
    if not 0 < n_internal_spatial < n_internal:
        raise ValueError(
            f'n_internal_spatial doit etre dans ]0, n_internal[ '
            f'(recu {n_internal_spatial} pour n_internal={n_internal}).'
        )

    n = n_sensory + n_internal + out_channels
    mask = torch.zeros(n, n)

    sens = torch.zeros(n_sensory, dtype=torch.bool)
    sens[torch.as_tensor(list(spatial_nodes), dtype=torch.long)] = True
    spatial_s = torch.arange(n_sensory)[sens]
    temporal_s = torch.arange(n_sensory)[~sens]

    int_spatial = torch.arange(n_sensory, n_sensory + n_internal_spatial)
    int_temporal = torch.arange(n_sensory + n_internal_spatial, n_sensory + n_internal)
    labels = torch.arange(n_sensory + n_internal, n)

    def connect(a, b):
        if len(a) == 0 or len(b) == 0:
            return
        mask[a.view(-1, 1), b.view(1, -1)] = 1.0
        mask[b.view(-1, 1), a.view(1, -1)] = 1.0

    connect(spatial_s, int_spatial)      # spatial <-> interne spatial
    connect(temporal_s, int_temporal)    # temporel <-> interne temporel
    connect(int_spatial, labels)         # les deux sous-blocs se rejoignent
    connect(int_temporal, labels)        # uniquement au niveau des labels

    mask.fill_diagonal_(0.0)
    return mask


class PCGraphCore(nn.Module):
    """Moteur generique de predictive coding sur graphe (Eq. 1-4 de l'article).

    mu_i = sum_j theta[j, i] * f(x_j)   (Eq. 1)
    eps_i = x_i - mu_i
    E = 1/2 * sum_i eps_i^2             (Eq. 2)

    `mask[j, i] = 1` autorise la connexion j -> i, `0` l'interdit. `theta` est
    multiplie par `mask` a chaque usage : les entrees masquees ont un gradient
    exactement nul, pas besoin de les remasquer apres chaque pas d'optimiseur.
    """

    def __init__(self, n: int, mask: torch.Tensor, init_std: float = 0.05):
        super().__init__()
        self.n = n
        self.register_buffer('mask', mask)
        self.theta = nn.Parameter(torch.randn(n, n) * init_std)
        self.act = nn.Hardtanh()

    def theta_masked(self) -> torch.Tensor:
        return self.theta * self.mask

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) @ self.theta_masked()

    def energy_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        eps = x - self.predict(x)
        return 0.5 * eps.pow(2).sum(dim=-1)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_per_sample(x).mean()

    def relax(self, x_init: torch.Tensor, clamp_mask: torch.Tensor, t_steps: int,
              lr_x: float, differentiable: bool = False) -> torch.Tensor:
        """Inference generique (Eq. 3) : les noeuds ou `clamp_mask` est True
        gardent leur valeur de `x_init` pour toute la duree (query by
        conditioning) ; les autres sont relaches par descente de gradient sur
        l'energie.

        `differentiable=False` (mode "phantom", par defaut) : chaque pas
        construit puis detruit un graphe d'autodiff local a x -- cout memoire
        O(1) par pas, independant de `t_steps`. Le resultat renvoye ne porte
        aucune dependance vis-a-vis de `theta` (cf. PCGraphModel.forward pour
        comment le gradient vers `theta` est ensuite obtenu en un seul pas
        supplementaire, a l'equilibre).

        `differentiable=True` (mode "bptt", recherche) : le graphe est
        conserve sur toute la relaxation (retropropagation a travers le temps
        complete) -- beaucoup plus couteux en memoire, utile seulement pour
        comparer a l'approximation "phantom" ci-dessus.
        """
        free = (~clamp_mask).float()

        if differentiable:
            # torch.enable_grad() force la construction du graphe meme si
            # l'appelant est sous torch.no_grad() (cas de l'evaluation/
            # prediction du pipeline, cf. Training.launch_val_test_loader) --
            # sans quoi `create_graph=True` echoue faute de grad_fn sur E. Si
            # l'appelant etait bien sous no_grad (contexte eval, aucun backward
            # ne sera fait), on redetache le resultat en sortie pour ne pas
            # laisser fuir un graphe inutile.
            was_grad_enabled = torch.is_grad_enabled()
            with torch.enable_grad():
                x = x_init.clone().requires_grad_(True)
                for _ in range(t_steps):
                    E = self.energy(x)
                    (grad,) = torch.autograd.grad(E, x, create_graph=True)
                    x = x - lr_x * grad * free
            return x if was_grad_enabled else x.detach()

        x = x_init.clone().detach().requires_grad_(True)
        for _ in range(t_steps):
            with torch.enable_grad():
                E = self.energy(x)
                (grad,) = torch.autograd.grad(E, x)
            with torch.no_grad():
                x = x - lr_x * grad * free
            x = x.detach().requires_grad_(True)
        return x.detach()

    @staticmethod
    def _prune_outgoing(m: torch.Tensor, prune_top_k: int = None) -> torch.Tensor:
        """Ne garde que les `prune_top_k` aretes sortantes les plus fortes de
        chaque noeud (cf. `causal_strength_matrix`). `None` = aucun elagage."""
        if prune_top_k is None or prune_top_k >= m.shape[1]:
            return m
        keep = torch.zeros_like(m)
        keep.scatter_(1, m.topk(prune_top_k, dim=1).indices, 1.0)
        return m * keep

    def causal_strength_matrix(self, n_hops: int = 3, prune_top_k: int = None) -> torch.Tensor:
        """Force d'influence i -> j cumulee sur `n_hops` sauts, a partir de
        `|theta_masked()|` -- utilise par les outils de visualisation pour
        classer les chemins sensoriel -> label sans avoir a re-executer de
        relaxation (juste une lecture des poids appris).

        Avec une topologie 'full' (tout est structurellement autorise), la
        matrice `|theta_masked()|` reste dense : chaque noeud a des centaines
        d'aretes sortantes non nulles, dont l'immense majorite proche de
        l'echelle d'initialisation (bruit, jamais vraiment utilisee par le
        reseau). Sommees sur `n_hops` sauts, ces nombreuses aretes faibles
        peuvent peser plus lourd au total que les quelques aretes fortes
        reellement significatives -- diluant le classement au lieu de le
        clarifier. `prune_top_k`, si fourni, ne garde que les `prune_top_k`
        aretes sortantes les plus fortes de CHAQUE noeud (les autres mises a
        zero) avant d'accumuler les sauts, pour calculer la force causale
        seulement sur le squelette structurellement significatif du graphe
        -- au prix de possiblement rater une influence reelle mais diffuse
        sur beaucoup d'aretes tres faibles. Par defaut (`None`) le calcul
        reste dense/exact, comme avant.
        """
        with torch.no_grad():
            m = self._prune_outgoing(self.theta_masked().abs(), prune_top_k)
            total = torch.zeros_like(m)
            power = m
            for _ in range(n_hops):
                total = total + power
                power = power @ m
        return total

    def causal_strength_mediated(self, start_idx, mid_idx, end_idx, n_hops: int = 3,
                                 prune_top_k: int = None) -> torch.Tensor:
        """Variante de `causal_strength_matrix` restreinte a une question
        precise : "quelle est la force d'influence de `start_idx` sur
        `end_idx`, en ne comptant QUE les chemins dont tous les noeuds
        INTERMEDIAIRES appartiennent a `mid_idx`" (ex: sensoriel -> interne
        -> ... -> interne -> label). `causal_strength_matrix` autorise
        n'importe quel noeud comme intermediaire -- pour une chaine
        sensoriel -> label, ca inclut des detours par D'AUTRES noeuds
        sensoriels (ex. Hetre_mean -> Hetre_max, une arete de reconstruction
        a 0.48, sans lien avec le label), qui mesurent alors une correlation
        feature<->feature plutot qu'une influence mediee par la
        representation interne du reseau -- verifie empiriquement : seuls
        4/15 features du classement se recouvrent entre les deux versions.

        Retourne une matrice `(len 1D range de start_idx, len de end_idx)`,
        pas la matrice `(n, n)` complete de `causal_strength_matrix`.
        """
        with torch.no_grad():
            m = self._prune_outgoing(self.theta_masked().abs(), prune_top_k)
            m_se = m[start_idx, end_idx]
            m_sm = m[start_idx, mid_idx]
            m_mm = m[mid_idx, mid_idx]
            m_me = m[mid_idx, end_idx]

            total = m_se.clone()          # 1 saut : arete directe
            acc = m_sm                    # start -> mid
            for _ in range(n_hops - 1):
                total = total + acc @ m_me
                acc = acc @ m_mm          # un intermediaire de plus
        return total

    def strongest_paths(self, start_idx, mid_idx, end_idx, max_hops: int = 3):
        """Le chemin UNIQUE le plus fort start -> [mid]* -> end, pour chaque
        (start, end), sur chaque longueur de 1 a `max_hops` sauts --
        contrairement a `causal_strength_mediated`, qui SOMME (en valeur
        absolue) sur TOUS les chemins possibles, ceci cherche le MEILLEUR
        chemin individuel et conserve son SIGNE reel (`theta_masked()`, pas
        `.abs()`). Utile pour repondre a "quelle est LA combinaison
        (feature + noeuds internes) qui explique le mieux ce label, et
        pousse-t-elle sa valeur vers le haut ou vers le bas ?" -- une somme
        de 1000 chemins faibles peut avoir la meme magnitude qu'un seul
        chemin fort, mais seul ce dernier est une explication lisible.

        Calcul par force brute (produits tensoriels), limite a `max_hops <=
        3` -- une version generale par programmation dynamique (Viterbi sur
        le graphe) serait necessaire au-dela, le cout croissant en
        `len(mid_idx) ** (max_hops - 1)`.

        Retourne une liste de `max_hops` `(value, mids)` -- un par longueur
        de chemin (1, 2, ..., max_hops) :
          - `value` : tenseur signe `(len(start_idx), len(end_idx))`, la
            valeur du meilleur chemin de CETTE longueur exacte pour chaque
            paire (start, end).
          - `mids` : `None` pour longueur 1 (chemin direct, pas
            d'intermediaire) ; sinon un tenseur entier
            `(len(start_idx), len(end_idx), longueur-1)` donnant les indices
            (locaux a `mid_idx`) des noeuds intermediaires traverses, dans
            l'ordre start -> mid[0] -> mid[1] -> ... -> end.
        """
        if max_hops > 3:
            raise NotImplementedError(
                'strongest_paths : max_hops > 3 non supporte (force brute, cout en '
                'len(mid_idx) ** (max_hops - 1) ; une version Viterbi serait necessaire).'
            )

        with torch.no_grad():
            theta = self.theta_masked()
            m_se = theta[start_idx, end_idx]          # (S, E)
            m_sm = theta[start_idx, mid_idx]          # (S, M)
            m_mm = theta[mid_idx, mid_idx]            # (M, M)
            m_me = theta[mid_idx, end_idx]            # (M, E)

            results = [(m_se.clone(), None)]

            if max_hops >= 2:
                # (S, M, E) : start -> i -> end
                cand = m_sm.unsqueeze(2) * m_me.unsqueeze(0)
                best = cand.abs().argmax(dim=1)                        # (S, E)
                value = torch.gather(cand, 1, best.unsqueeze(1)).squeeze(1)
                results.append((value, best.unsqueeze(-1)))

            if max_hops >= 3:
                S, M = m_sm.shape
                E = m_me.shape[1]
                # (S, M, M, E) : start -> i -> j -> end
                cand = (m_sm.view(S, M, 1, 1)
                        * m_mm.view(1, M, M, 1)
                        * m_me.view(1, 1, M, E))
                flat = cand.reshape(S, M * M, E)
                best = flat.abs().argmax(dim=1)                        # (S, E)
                value = torch.gather(flat, 1, best.unsqueeze(1)).squeeze(1)
                mids = torch.stack([torch.div(best, M, rounding_mode='floor'),
                                    best % M], dim=-1)                 # (S, E, 2)
                results.append((value, mids))

        return results


class PCGraphModel(nn.Module):
    """Enveloppe `PCGraphCore` au contrat attendu par le pipeline :
    `forward(x, z_prev=None) -> (output, logits, hidden)`.

    Les noeuds sont disposes en trois blocs contigus -- sensoriel (les
    features, `in_dim * seq_len` noeuds), interne (`n_internal`), puis label
    (`out_channels`, ou 1 en `label_mode='clm'`). Les slices correspondantes
    sont exposees pour l'outillage d'analyse.
    """

    def __init__(self, in_dim: int, k_days: int, out_channels: int, task_type: str,
                 n_internal: int = 128, t_train: int = 20, t_query: int = 50,
                 lr_x: float = 0.5, init_std: float = 0.05,
                 grad_mode: str = 'phantom', topology: str = 'full',
                 device: str = 'cpu', horizon: int = 0,
                 spatial_feature_idx=None, n_internal_spatial: int = 16,
                 label_mode: str = 'onehot', clm_tau: float = 0.3,
                 **_ignored):
        super().__init__()

        if task_type == 'uclassification':
            out_channels += 1

        self.in_dim = in_dim
        self.seq_len = k_days + 1
        self.out_channels = out_channels
        self.task_type = task_type
        self.n_internal = n_internal
        self.t_train = t_train
        self.t_query = t_query
        self.lr_x = lr_x
        self.grad_mode = grad_mode
        self.horizon = horizon
        self.device = device
        self.return_hidden = True

        # Nombre de NOEUDS de label, distinct du nombre de CLASSES de sortie.
        #
        # 'onehot' : K noeuds, un par classe.
        # 'corn'   : K noeuds = [s, K-1 logits conditionnels].
        # 'clm'    : UN seul noeud, le score de risque scalaire `s`. Les classes
        #            viennent de seuils appliques a `s` -- il n'y a donc rien
        #            d'autre a representer.
        #
        # Pourquoi 'clm' plutot que 'corn' : dans CORN, P(y=k) = F_{k-1} - F_k
        # avec F un produit cumule. Les classes intermediaires n'heritent que
        # des petites differences d'une suite qui decroit geometriquement, la
        # derniere encaissant tout le reste -- donc p_0 > p_1 > p_2 > p_3 PAR
        # CONSTRUCTION, et l'argmax ne peut valoir que 0 ou K-1. Verifie en
        # simulation (2 classes utilisees sur 5) et en entrainement reel
        # (distribution predite [1483, 0, 0, 1536, 2821], classes 1 et 2 vides,
        # loss de couverture bloquee a son maximum sans qu'aucun reglage de
        # lambda puisse y remedier).
        #
        # Le CLM applique au contraire des seuils ORDONNES au MEME score :
        # p_k = sigmoid(theta_k - s) - sigmoid(theta_{k-1} - s) est une bosse,
        # et l'argmax selectionne l'intervalle contenant `s`. Les classes
        # intermediaires redeviennent atteignables.
        if label_mode not in ('onehot', 'corn', 'clm'):
            raise ValueError(
                f"label_mode doit valoir 'onehot', 'corn' ou 'clm' (recu {label_mode!r})")
        self.label_mode = label_mode
        n_label = 1 if label_mode == 'clm' else out_channels
        self.n_label = n_label

        n_sensory = in_dim * self.seq_len
        self.n_sensory = n_sensory
        self.sensory_slice = slice(0, n_sensory)
        self.internal_slice = slice(n_sensory, n_sensory + n_internal)
        self.label_slice = slice(n_sensory + n_internal, n_sensory + n_internal + n_label)
        n = n_sensory + n_internal + n_label
        self.n_nodes = n

        lab0 = n_sensory + n_internal
        self.scalar_slice = slice(lab0, lab0 + 1)                 # `s` dans tous les modes
        self.corn_slice = slice(lab0 + 1, lab0 + n_label)         # vide si clm

        # Seuils du CLM : APPRIS.
        #
        # Les figer aux quantiles de la distribution cible rendait le critere de
        # couverture vide : un `s` de BRUIT PUR standardise, coupe a ces
        # quantiles, redonne deja la distribution voulue ([5633, 3725, 672, 352,
        # 69] contre une cible [5539, 3687, 849, 321, 55]). Le modele obtenait
        # donc la bonne couverture sans rien apprendre, et perdait toute
        # expressivite sur la distribution -- impossible d'exprimer "cet ete est
        # sec, beaucoup de jours a risque 4", le binning par quantiles forcant
        # 0.5% des jours en classe 4 en toute saison.
        #
        # L'objection qui les faisait figer ("aucun gradient depuis l'energie")
        # ne tient plus : `p_k` depend de theta via sigmoid((theta_k - s)/tau),
        # donc les termes distributionnels leur en fournissent un. Chaque piece
        # retrouve alors son role : `s` est contraint par l'ordinalite, les
        # seuils par la couverture, et aucun n'est satisfait gratuitement.
        #
        # Parametrage en increments positifs : theta_k = theta_0 + somme des
        # softplus(delta_j). La stricte croissance -- exigee sous peine de
        # probabilites negatives -- est ainsi garantie PAR CONSTRUCTION, sans
        # projection ni contrainte a maintenir pendant l'optimisation.
        self.clm_min_gap = 1e-2
        self.clm_base = nn.Parameter(torch.zeros(1))
        self.clm_deltas = nn.Parameter(torch.zeros(max(out_channels - 2, 0)))
        self.clm_tau = float(clm_tau)

        # Mise a l'echelle de `s` : division par une CONSTANTE, PAS par son
        # propre ecart-type.
        #
        # Une standardisation (retrancher la moyenne, diviser par l'ecart-type
        # courant) DETRUIT l'information d'echelle : si `s` retrecit, le
        # denominateur retrecit avec lui et le quotient est inchange. Verifie --
        # `s` multiplie par 10 000, `L_ord` et `L_cov` restaient identiques au
        # septieme chiffre. La loss devenait donc aveugle a l'amplitude, tandis
        # que le weight decay la reduisait librement : `s` s'ecrasait sans
        # penalite (ecart-type 0.205 sur le train, 0.016 sur des dates neuves)
        # et tout le test tombait dans une seule classe.
        #
        # Avec un diviseur CONSTANT, l'echelle est preservee et la loss voit
        # l'effondrement : un `s` ecrase de 100x fait grimper `L_cov` d'un
        # facteur 11. C'est `clm_tau`, fixe, qui sert d'ancre -- si `s` et les
        # seuils retrecissent ensemble, `(theta_k - s)/tau` retrecit aussi, les
        # bosses se recouvrent et la couverture se degrade.
        #
        # Renseigne par le Training via `set_clm_scale` (ecart-type de la cible
        # reelle). Purement numerique : amener `s` dans une plage saine, jamais
        # absorber ses variations d'amplitude.
        self.register_buffer('clm_scale', torch.ones(1))

        if topology == 'full':
            mask = fully_connected_mask(n)
        elif topology == 'layered':
            mask = layered_mask([n_sensory, n_internal, n_label])
        elif topology == 'bimodal':
            if spatial_feature_idx is None:
                raise ValueError(
                    "topology='bimodal' exige `spatial_feature_idx` (indices des features "
                    "spatiales, au niveau FEATURE et non noeud) -- il est calcule depuis "
                    '`features_name` par PCGraphTraining.make_model.'
                )
            # Une feature occupe `seq_len` noeuds consecutifs (f*T + t) : on
            # etend les indices feature -> indices noeud avant de masquer.
            T = self.seq_len
            spatial_nodes = [f * T + t for f in spatial_feature_idx for t in range(T)]
            mask = bimodal_mask(n_sensory, n_internal, n_label,
                                spatial_nodes, n_internal_spatial)
            self.n_internal_spatial = n_internal_spatial
            self.internal_spatial_slice = slice(n_sensory, n_sensory + n_internal_spatial)
            self.internal_temporal_slice = slice(n_sensory + n_internal_spatial,
                                                 n_sensory + n_internal)
        else:
            raise ValueError(f'Unknown PCGraph topology: {topology}')

        self.core = PCGraphCore(n, mask, init_std=init_std)

        clamp = torch.zeros(n, dtype=torch.bool)
        clamp[self.sensory_slice] = True
        self.register_buffer('clamp_mask', clamp)

        # Masque d'entrainement (Algorithme 1 litteral) : sensoriel *et*
        # label clampes pour toute la relaxation, seuls les noeuds internes
        # sont relaches -- distinct de `clamp_mask` (sensoriel seul) qui sert
        # a l'inference/prediction (label libre = ce qu'on veut predire).
        clamp_train = torch.zeros(n, dtype=torch.bool)
        clamp_train[self.sensory_slice] = True
        clamp_train[self.label_slice] = True
        self.register_buffer('clamp_mask_train', clamp_train)

        if task_type in ('classification', 'binary'):
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor, z_prev: torch.Tensor = None):
        B = x.shape[0]
        x_sensory = x.reshape(B, -1)
        if x_sensory.shape[1] != self.n_sensory:
            raise ValueError(
                f'PCGraphModel: attendu {self.n_sensory} valeurs sensorielles '
                f'(in_dim={self.in_dim} x seq_len={self.seq_len}), recu {x_sensory.shape[1]}'
            )

        x_init = x_sensory.new_zeros(B, self.n_nodes)
        x_init[:, self.sensory_slice] = x_sensory
        if z_prev is not None and z_prev.dim() == 2 and z_prev.shape[1] == self.n_internal:
            x_init[:, self.internal_slice] = z_prev.detach()

        clamp_mask = self.clamp_mask
        t_steps = self.t_train if self.training else self.t_query
        differentiable = self.grad_mode == 'bptt'

        x_conv = self.core.relax(x_init, clamp_mask, t_steps, self.lr_x,
                                 differentiable=differentiable)

        if differentiable:
            node_values = x_conv
        else:
            # Point fixe detache (x_conv independant de theta) : un unique pas
            # de prediction differentiable reattache le gradient vers `theta`
            # -- approximation "phantom gradient" du gradient implicite au
            # point fixe (Geng et al., 2021), cout memoire O(1) au lieu de
            # O(t_steps) pour une retropropagation complete.
            node_values = self.core.predict(x_conv)

        logits = node_values[:, self.label_slice]
        hidden = x_conv[:, self.internal_slice]
        output = self.class_probs(logits)

        return output, logits, hidden

    @property
    def clm_thresholds(self) -> torch.Tensor:
        """Seuils effectifs, croissants par construction."""
        if self.clm_deltas.numel() == 0:
            return self.clm_base
        # `+ clm_min_gap` : softplus tend vers 0 pour un delta tres negatif, ce
        # qui collerait deux seuils et viderait la classe intermediaire. L'ecart
        # minimal garantit une croissance STRICTE quels que soient les
        # parametres, sans contrainte a maintenir pendant l'optimisation.
        gaps = nn.functional.softplus(self.clm_deltas) + self.clm_min_gap
        return torch.cat([self.clm_base, self.clm_base + torch.cumsum(gaps, 0)])

    def set_clm_scale(self, scale):
        """Constante de mise a l'echelle de `s` (typiquement l'ecart-type de la
        cible reelle). Fixe : elle amene `s` dans une plage numerique saine sans
        jamais absorber ses variations d'amplitude."""
        with torch.no_grad():
            self.clm_scale.fill_(max(float(scale), 1e-6))

    def set_clm_thresholds(self, thresholds):
        """INITIALISE les seuils (ils restent appris ensuite). Les quantiles de
        la distribution cible sont un point de depart raisonnable, plus la
        reponse figee qu'ils etaient auparavant."""
        t = torch.as_tensor(thresholds, dtype=self.clm_base.dtype).view(-1)
        if t.numel() != self.out_channels - 1:
            raise ValueError(f'{self.out_channels - 1} seuils attendus, recu {t.numel()}')
        if not bool((t[1:] > t[:-1]).all()):
            raise ValueError(f'seuils non croissants : {t.tolist()}')
        with torch.no_grad():
            self.clm_base.copy_(t[:1])
            gaps = (t[1:] - t[:-1] - self.clm_min_gap).clamp_min(1e-4)
            self.clm_deltas.copy_(torch.log(torch.expm1(gaps)))   # softplus^-1

    def class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Probabilites de classe (B, out_channels) a partir des valeurs des
        noeuds de label. Point unique de lecture, partage par `forward` et par
        les termes distributionnels de l'entrainement -- pour qu'ils voient
        exactement la meme distribution."""
        if self.label_mode == 'clm':
            # p_k = sigmoid(theta_k - s) - sigmoid(theta_{k-1} - s) : une BOSSE
            # centree sur l'intervalle de seuils contenant `s`. Les seuils etant
            # ordonnes et espaces, les classes intermediaires sont atteignables
            # par l'argmax -- ce que le produit cumule de CORN interdisait.
            s = logits[:, 0:1]
            # Constante : identique en entrainement et a l'inference, aucune
            # statistique a maintenir, et surtout l'amplitude de `s` reste
            # visible par la loss (cf. commentaire sur `clm_scale`).
            s = s / self.clm_scale
            # La TEMPERATURE est critique, pas cosmetique : la sigmoide standard
            # (tau=1) transitionne sur ~4 unites alors que les seuils sont
            # espaces de ~0.6-0.85. Les bosses se recouvrent alors totalement et
            # l'argmax ne vaut plus que 0 ou K-1 -- exactement le defaut de CORN.
            # Verifie : tau=1 -> 2 classes atteignables, tau <= 0.5 -> les 5.
            F = torch.sigmoid((self.clm_thresholds.view(1, -1) - s) / self.clm_tau)
            first = F[:, :1]
            middle = F[:, 1:] - F[:, :-1]
            last = 1.0 - F[:, -1:]
            return torch.cat([first, middle, last], dim=1).clamp_min(0.0)
        if self.label_mode != 'corn':
            return self.output_activation(logits)
        # Import paresseux : `utils` tire torch_geometric/pytorch_tcn, alors que
        # ce module reste volontairement leger et testable en isolation.
        from forecasting_models.pytorch.utils import corn_class_probs
        return corn_class_probs(logits[:, 1:])

    def training_energy(self, x: torch.Tensor, y_clamp: torch.Tensor) -> torch.Tensor:
        """Signal d'apprentissage natif du PC-graph (Algorithme 1, Eq. 2-4) :
        sensoriel *et* label sont clampes (`clamp_mask_train`), seuls les
        noeuds internes sont relaches ; on renvoie l'energie par echantillon
        au point fixe. Contrairement a `forward`, il n'y a **aucune loss
        externe** ici -- l'energie elle-meme est ce qu'on retropropage vers
        `theta` (une seule mise a jour de poids, comme dans l'article),
        exactement l'usage attendu par PCGraphTraining.launch_batch."""
        B = x.shape[0]
        x_sensory = x.reshape(B, -1)
        x_init = x_sensory.new_zeros(B, self.n_nodes)
        x_init[:, self.sensory_slice] = x_sensory
        x_init[:, self.label_slice] = y_clamp

        t_steps = self.t_train if self.training else self.t_query
        differentiable = self.grad_mode == 'bptt'
        x_conv = self.core.relax(x_init, self.clamp_mask_train, t_steps, self.lr_x,
                                 differentiable=differentiable)

        # Point fixe detache en mode "phantom" (cf. forward) : recalculer
        # l'energie ici reattache le gradient vers `theta` sans BPTT.
        return self.core.energy_per_sample(x_conv)

    def training_energy_semi_supervised(self, x: torch.Tensor, y_clamp: torch.Tensor,
                                        label_clamp_prob: float = 0.7):
        """Variante "dropout de supervision" de `training_energy`, pour les
        cibles bruitees (cf. PCGraphGenTraining) : au lieu de clamper le
        label sur *tous* les echantillons du batch, chaque echantillon a une
        probabilite `label_clamp_prob` de voir son label clampe (comme
        `training_energy`) et sinon reste libre pendant toute la relaxation,
        exactement comme un noeud de label a l'inference (`forward`) -- son
        energie au point fixe ne compare alors jamais ce noeud a une valeur
        de target potentiellement fausse (un label a 0 peut en realite
        indiquer un risque eleve). `clamp_mask_train` (n,) est donc etendu
        par-echantillon (B, n) : `relax`/`energy_per_sample` restent
        inchanges, un masque (B, n) se diffuse element-par-element exactement
        comme le masque (n,) global qu'ils recoivent d'habitude.

        Retourne `(energy_per_sample, label_clamped)` -- `label_clamped`
        (bool, (B,)) est expose pour logging/diagnostic (fraction supervisee
        du batch), sans effet sur l'entrainement lui-meme."""
        B = x.shape[0]
        x_sensory = x.reshape(B, -1)
        x_init = x_sensory.new_zeros(B, self.n_nodes)
        x_init[:, self.sensory_slice] = x_sensory

        label_clamped = torch.rand(B, device=x.device) < label_clamp_prob
        x_init[:, self.label_slice] = y_clamp * label_clamped.unsqueeze(1).float()

        clamp_mask = self.clamp_mask_train.unsqueeze(0).expand(B, -1).clone()
        clamp_mask[:, self.label_slice] &= label_clamped.unsqueeze(1)

        t_steps = self.t_train if self.training else self.t_query
        differentiable = self.grad_mode == 'bptt'
        x_conv = self.core.relax(x_init, clamp_mask, t_steps, self.lr_x,
                                 differentiable=differentiable)

        return self.core.energy_per_sample(x_conv), label_clamped
