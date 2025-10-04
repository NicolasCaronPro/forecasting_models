from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

class WKLoss(nn.Module):
    """
    Implements Weighted Kappa Loss, introduced by :footcite:t:`deLaTorre2018kappa`.
    Weighted Kappa is widely used in ordinal classification problems. Its value lies in
    :math:`[0, 2]`, where :math:`2` means the random prediction.

    The loss is computed as follows:

    .. math::
        \\mathcal{L}(X, \\mathbf{y}) =
        \\frac{\\sum\\limits_{i=1}^J \\sum\\limits_{j=1}^J \\omega_{i,j}
        \\sum\\limits_{k=1}^N q_{k,i} ~ p_{y_k,j}}
        {\\frac{1}{N}\\sum\\limits_{i=1}^J \\sum\\limits_{j=1}^J \\omega_{i,j}
        \\left( \\sum\\limits_{k=1}^N q_{k,i} \\right)
        \\left( \\sum\\limits_{k=1}^N p_{y_k, j} \\right)}

    where :math:`q_{k,j}` denotes the normalised predicted probability, computed as:

    .. math::
        q_{k,j} = \\frac{\\text{P}(\\text{y} = j ~|~ \\mathbf{x}_k)}
        {\\sum\\limits_{i=1}^J \\text{P}(\\text{y} = i ~|~ \\mathbf{x}_k)},

    :math:`p_{y_k,j}` is the :math:`j`-th element of the one-hot encoded true label
    for sample :math:`k`, and :math:`\\omega` is the penalisation matrix, defined
    either linearly or quadratically. Its elements are:

    - Linear: :math:`\\omega_{i,j} = \\frac{|i - j|}{J - 1}`
    - Quadratic: :math:`\\omega_{i,j} = \\frac{(i - j)^2}{(J - 1)^2}`

    Parameters
    ----------
    num_classes : int
        The number of unique classes in your dataset.
    penalization_type : str, default='quadratic'
        The penalization method for calculating the Kappa statistics. Valid options are
        ``['linear', 'quadratic']``. Defaults to 'quadratic'.
    epsilon : float, default=1e-10
        Small value added to the denominator division by zero.
    weight : Optional[torch.Tensor], default=None
        Class weights to apply during loss computation. Should be a tensor of size
        `(num_classes,)`. If `None`, equal weight is given to all classes.
    use_logits : bool, default=False
        If `True`, the `input` is treated as logits. If `False`, `input` is treated
        as probabilities. The behavior of the `input` affects its expected format
        (logits vs. probabilities).

    Example
    -------
    >>> import torch
    >>> from dlordinal.losses import WKLoss
    >>> num_classes = 5
    >>> input = torch.randn(3, num_classes)  # Predicted logits for 3 samples
    >>> target = torch.randint(0, num_classes, (3,))  # Ground truth class indices
    >>> loss_fn = WKLoss(num_classes)
    >>> loss = loss_fn(input, target)
    >>> print(loss)
    """

    num_classes: int
    penalization_type: str
    weight: Optional[torch.Tensor]
    epsilon: float
    use_logits: bool

    def __init__(
        self,
        num_classes: int,
        penalization_type: str = "quadratic",
        weight: Optional[torch.Tensor] = None,
        epsilon: Optional[float] = 1e-10,
        use_logits=True,
    ):
        super(WKLoss, self).__init__()
        self.num_classes = num_classes
        self.penalization_type = penalization_type
        self.epsilon = epsilon
        self.weight = weight
        self.use_logits = use_logits
        self.first_forward_ = True

    def _initialize(self, input, target):
        # Define error weights matrix
        repeat_op = (
            torch.arange(self.num_classes, device=input.device)
            .unsqueeze(1)
            .expand(self.num_classes, self.num_classes)
        )
        if self.penalization_type == "linear":
            self.weights_ = torch.abs(repeat_op - repeat_op.T) / (self.num_classes - 1)
        elif self.penalization_type == "quadratic":
            self.weights_ = torch.square((repeat_op - repeat_op.T)) / (
                (self.num_classes - 1) ** 2
            )

        # Apply class weight
        if self.weight is not None:
            # Repeat weight num_classes times in columns
            tiled_weight = self.weight.repeat((self.num_classes, 1)).to(input.device)
            self.weights_ *= tiled_weight

    def forward(self, input, target):
        """
        Forward pass for the Weighted Kappa loss.

        This method computes the Weighted Kappa loss between the predicted and true labels.
        The loss is based on the weighted disagreement between predictions and true labels,
        normalised by the expected disagreement under independence.

        Parameters
        ----------
        input : torch.Tensor
            The model predictions. Shape: ``(batch_size, num_classes)``.
            If ``use_logits=True``, these should be raw logits (unnormalised scores).
            If ``use_logits=False``, these should be probabilities (rows summing to 1).

        target : torch.Tensor
            Ground truth labels.
            Shape:
            - ``(batch_size,)`` if labels are class indices.
            - ``(batch_size, num_classes)`` if already one-hot encoded.
            The tensor will be converted to float internally.

        Returns
        -------
        loss : torch.Tensor
            A scalar tensor representing the weighted disagreement between predictions
            and true labels, normalised by the expected disagreement.
        """

        num_classes = self.num_classes

        # Convert to onehot if integer labels are provided
        if target.dim() == 1:
            y = torch.eye(num_classes).to(target.device)
            target = y[target]

        target = target.float()

        if self.first_forward_:
            if not self.use_logits and not torch.allclose(
                input.sum(dim=1), torch.tensor(1.0, device=input.device)
            ):
                raise ValueError(
                    "When passing use_logits=False, the input"
                    " should be probabilities, not logits."
                )
            elif self.use_logits and torch.allclose(
                input.sum(dim=1), torch.tensor(1.0, device=input.device)
            ):
                raise ValueError(
                    "When passing use_logits=True, the input"
                    " should be logits, not probabilities."
                )

            self._initialize(input, target)
            self.first_forward_ = False

        if self.use_logits:
            input = torch.nn.functional.softmax(input, dim=1)

        hist_rater_a = torch.sum(input, 0)
        hist_rater_b = torch.sum(target, 0)

        conf_mat = torch.matmul(input.T, target)

        bsize = input.size(0)
        nom = torch.sum(self.weights_ * conf_mat)
        expected_probs = torch.matmul(
            torch.reshape(hist_rater_a, [num_classes, 1]),
            torch.reshape(hist_rater_b, [1, num_classes]),
        )
        denom = torch.sum(self.weights_ * expected_probs / bsize)

        return nom / (denom + self.epsilon)

class MCELoss(torch.nn.modules.loss._WeightedLoss):
    """
    Mean Squared Error (MSE) loss computed per class. This loss function calculates the
    MSE for each class independently and then reduces it based on the specified `reduction`
    method. It is useful in scenarios where each class needs to be treated independently
    during the loss computation.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification problem.

    weight : Optional[Tensor], default=None
        A tensor of size `J`, where `J` is the number of classes, representing the weight
        for each class. If provided, each class's MSE will be scaled by its corresponding
        weight. If not provided, all classes are treated with equal weight (i.e., all weights
        are set to 1).

    reduction : str, default='mean'
        The method to reduce the MSE values across all classes:
        - `'none'`: No reduction is applied. A tensor of MSE values for each class is returned.
        - `'mean'`: The mean of the MSE values across all classes is returned.
        - `'sum'`: The sum of the MSE values across all classes is returned.

    use_logits : bool, default=False
        If True, the `input` tensor (predictions) is assumed to be in logits format. If False,
        the `input` tensor is treated as probabilities.

    Example
    -------
    >>> import torch
    >>> from torch.nn import CrossEntropyLoss
    >>> from dlordinal.losses import MCELoss
    >>> num_classes = 5
    >>> base_loss = CrossEntropyLoss()
    >>> loss = MCELoss(num_classes=num_classes)
    >>> input = torch.randn(3, num_classes)
    >>> target = torch.randint(0, num_classes, (3,))
    >>> output = loss(input, target)

    Notes
    -----
    - The class supports both the use of logits and probabilities in the predictions.
    - When `use_logits=True`, the input is passed through a softmax function before computing
      the MSE. If `use_logits=False`, the `input` tensor is expected to already contain
      probabilities.
    """

    def __init__(
        self,
        num_classes: int,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=False,
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

    def compute_per_class_mse(self, input: torch.Tensor, target: torch.Tensor):
        """
        Computes the mean squared error (MSE) for each class independently.

        Parameters
        ----------
        input : torch.Tensor
            Predicted labels (either logits or probabilities, depending on `use_logits`).

        target : torch.Tensor
            Ground truth labels in one-hot encoding format.

        Returns
        -------
        mses : torch.Tensor
            A tensor containing the MSE values for each class.
        """

        if input.shape != target.shape:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        if input.shape != target.shape:
            raise ValueError(
                f"Input shape {input.shape} is not compatible with target shape "
                + f"{target.shape}"
            )

        if self.use_logits:
            input = torch.nn.functional.softmax(input, dim=1)

        # Compute the squared error for each class
        per_class_se = torch.pow(target - input, 2)

        # Apply class weights if defined
        if self.weight is not None:
            tiled_weight = torch.tile(self.weight, (per_class_se.shape[0], 1))
            per_class_se = per_class_se * tiled_weight

        # Compute the mean squared error for each class
        per_class_mse = torch.mean(per_class_se, dim=0)

        return per_class_mse

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        input : torch.Tensor
            Predicted labels. Should be logits if `use_logits` is True, otherwise
            probabilities.

        target : torch.Tensor
            Ground truth labels, typically in class indices.

        Returns
        -------
        reduced_mse : torch.Tensor
            The MSE per class reduced using the specified `reduction` method. If
            `reduction='none'`, the MSE values for each class are returned.
            Otherwise, the MSE is reduced according to the method (`mean`, `sum`).
        """

        target_oh = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        per_class_mse = self.compute_per_class_mse(input, target_oh)

        if self.reduction == "mean":
            reduced_mse = torch.mean(per_class_mse)
        elif self.reduction == "sum":
            reduced_mse = torch.sum(per_class_mse)
        else:
            reduced_mse = per_class_mse

        return reduced_mse