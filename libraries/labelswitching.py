#
import numpy as np
import torch
import torch.nn as nn

from libraries.ML_models import MLPClassifierTorch, LogisticRegressionTorch


import random
from imblearn.over_sampling import SMOTE


from scipy.optimize import fmin_l_bfgs_b
from torch.optim import Optimizer
from functools import reduce

# from sklearn.metrics import log_loss

# debug = False # True # 
# if debug:
#     import pdb

eps=np.finfo(float).eps

# from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler, TensorDataset


class LBFGSScipy(Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now CPU only
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size)
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self._n_iter = 0
        self._last_loss = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            self._distribute_flat_params(flat_params)
            loss = closure()
            self._last_loss = loss
            loss = loss.data
            flat_grad = self._gather_flat_grad().numpy()
            return loss, flat_grad

        def callback(flat_params):
            self._n_iter += 1

        initial_params = self._gather_flat_params()

        
        fmin_l_bfgs_b(wrapped_closure, initial_params, maxiter=max_iter,
                      maxfun=max_eval,
                      factr=tolerance_change / eps, pgtol=tolerance_grad, epsilon=1e-08,
                      m=history_size,
                      callback=callback)

def label_switching(y, w=None, alphasw=0.0, betasw=0.0):
    """
    Perform label switching with optional weight adjustment for sample-level weights.

    Parameters:
    - y: Original labels (numpy array with values -1 and +1).
    - w: Class weights (list or numpy array of size 2, [majority_weight, minority_weight]).
         If None, weights are initialized to [1.0, 1.0].
    - alphasw: Label switching rate from Majority to Minority class.
    - betasw: Label switching rate from Minority to Majority class.

    Returns:
    - ysw: Labels after switching.
    - wsw: Updated sample weights after switching.
    """
    # Initialize class weights if not provided
    if w is None:
        w = np.array([1.0, 1.0])  # [majority_class_weight, minority_class_weight]

    # Copy original labels
    ysw = np.copy(y)

    # Initialize sample weights based on the original labels
    wsw = np.where(y == -1, w[0], w[1])

    # Find indices of each class
    idx1 = np.where(y == +1)[0]  # Minority class
    l1 = len(idx1)
    bet_1 = int(round(l1 * betasw))  # Number of switches from Minority to Majority
    bet_1 = min(bet_1, l1)  # Ensure bet_1 <= l1
    if bet_1 > 0:
        idx1_sw = np.random.choice(idx1, bet_1, replace=False)
        # Perform label switching for the minority class
        ysw[idx1_sw] = -1
        wsw[idx1_sw] = w[0]  # Update weights to majority class weight

    idx0 = np.where(y == -1)[0]  # Majority class
    l0 = len(idx0)
    alph_0 = int(round(l0 * alphasw))  # Number of switches from Majority to Minority
    alph_0 = min(alph_0, l0)  # Ensure alph_0 <= l0
    if alph_0 > 0:
        idx0_sw = np.random.choice(idx0, alph_0, replace=False)
        # Perform label switching for the majority class
        ysw[idx0_sw] = +1
        wsw[idx0_sw] = w[1]  # Update weights to minority class weight

    return ysw, wsw

def compute_weights(targets_train, RB = 1, IR = 1, mode = 'Normal'):
    # RB define la cantidad de reequilibrado final
    # Si RB = IR => No se reequilibra. Si RB = 1, es un reequilibrado full.
    
    weights = np.ones_like(targets_train) # .astype('float')
    if mode == 'Small':
        weights[np.where(targets_train<=0)[0]] = RB/IR
    else:
        weights[np.where(targets_train>0)[0]] = RB # IR/RB
    
    return torch.from_numpy(weights) # /np.sum(weights))


from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def generate_batches(nBatch, param, mode='random', seed=42):
    if seed is not None:
        np.random.seed(seed)

    if mode == 'random':
        # param: number of samples in the training set
        if nBatch == param:
            list_samples_batch = []
            list_samples_batch.append(list(range(param)))
            num_batches = 1

        else:

            indices = list(range(param))
            random.shuffle(indices)

            l = len(indices)
            #for ndx in range(0, l, nBatch):
            #    yield indices[ndx:min(ndx + nBatch, l)]

            num_batches = int(np.ceil(param/nBatch))
            list_samples_batch = []
            for ndx in range(0, l, nBatch):
                list_aux = indices[ndx:min(ndx + nBatch, l)]
                list_aux.sort()
                list_samples_batch.append(list_aux)


    elif mode == 'class_equitative':
        # All classes have the same number of samples in each batch (repetition for minority)
        #param: class labels for the train set
        if nBatch == param.shape[0]:
            list_samples_batch = []
            list_samples_batch.append(list(range(param.shape[0])))
            num_batches = 1

        else:
            class_labels, samples_class = np.unique(param, return_counts=True)
            samples_max = np.max(samples_class)
            samples_min = np.min(samples_class)
            num_classes = class_labels.shape[0]

            batch_samples_class = np.ceil(nBatch/num_classes).astype(int)
            num_batches = np.ceil(samples_max/batch_samples_class).astype(int)

            list_samples_class = []
            for k in range(num_classes):
                mod = int((batch_samples_class*num_batches) // samples_class[k])
                rem = int((batch_samples_class*num_batches) % samples_class[k])
                ind_class = np.nonzero(param==class_labels[k])[0]
                list_aux = list(ind_class)*mod + list(np.random.choice(ind_class, rem))
                random.shuffle(list_aux)
                list_samples_class.append(list_aux)

            list_samples_batch = []
            for kbatch in range(num_batches):
                list_aux = []
                for kclass in range(num_classes):
                    list_aux += list_samples_class[kclass][batch_samples_class*kbatch:batch_samples_class*(kbatch+1)]

                list_aux.sort()
                list_samples_batch.append(list_aux)

    elif mode == 'representative':
        # All classes have at least 1 sample in each batch (repetition for minority)
        #param: class labels for the train set
        if nBatch == param.shape[0]:
            list_samples_batch = []
            list_samples_batch.append(list(range(param.shape[0])))
            num_batches = 1

        else:
            class_labels, samples_class = np.unique(param, return_counts=True)
            samples_max = np.max(samples_class)
            samples_min = np.min(samples_class)
            num_classes = class_labels.shape[0]
            num_samples = param.shape[0]

            #batch_samples_class = np.ceil(nBatch/num_classes).astype(int)
            num_batches = np.ceil(num_samples/nBatch).astype(int)
            batch_samples_class = []
            list_samples_class = []
            for k in range(num_classes):
                batch_samples_class.append(np.ceil(nBatch*samples_class[k]/num_samples).astype(int))

                mod = int((batch_samples_class[k]*num_batches) // samples_class[k])
                rem = int((batch_samples_class[k]*num_batches) % samples_class[k])
                ind_class = np.nonzero(param==class_labels[k])[0]
                list_aux = list(ind_class)*mod + list(np.random.choice(ind_class, rem))
                random.shuffle(list_aux)
                list_samples_class.append(list_aux)

            list_samples_batch = []
            for kbatch in range(num_batches):
                list_aux = []
                for kclass in range(num_classes):
                    list_aux += list_samples_class[kclass][batch_samples_class[kclass]*kbatch:batch_samples_class[kclass]*(kbatch+1)]

                list_aux.sort()
                list_samples_batch.append(list_aux)


    for kbatch in range(num_batches):
        yield list_samples_batch[kbatch]

def create_dataloader(X, y, weights, batch_size, mode='random', seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = TensorDataset(X, y, weights)

    # Generate indices using the original generate_batches (NumPy)
    batch_indices_orig = list(generate_batches(batch_size, y.numpy() if mode != "random" else len(y), mode, seed))
    batch_indices_orig = [sorted(batch) for batch in batch_indices_orig]  # Sort indices

    # Create a custom sampler using the generated indices
    from torch.utils.data import Sampler

    class BatchSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __iter__(self):
            for batch in self.indices:
                yield torch.tensor(batch)  # Yield PyTorch tensors

    batch_sampler = BatchSampler(batch_indices_orig)

    # Create the DataLoader using the custom sampler
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    return dataloader


def train_model(x, y, model, loss_fn, optimizer, weights, num_epochs=10, batch_size=None, mode='random', lbfgs=False, debug=False):
    """
    Train a model using either gradient-based optimizers or LBFGS.
    
    Parameters:
    - x: Features (torch.Tensor).
    - y: Labels (torch.Tensor).
    - model: PyTorch model to train.
    - loss_fn: Loss function (callable).
    - optimizer: Optimizer instance (torch.optim or LBFGSScipy).
    - weights: Sample weights (torch.Tensor).
    - num_epochs: Maximum number of epochs (int).
    - batch_size: Batch size for DataLoader (int).
    - mode: Sampling mode for DataLoader ('random', 'class_equitative', etc.).
    - lbfgs: If True, use LBFGS optimization logic (bool).
    - debug: If True, print debug information.
    """
    
    batch_size = len(x) if batch_size is None or batch_size == 'auto' else int(batch_size)

    trainloader = create_dataloader(x, y, weights, batch_size=batch_size, mode=mode)

    for epoch in range(num_epochs):
        
        """
        if not lbfgs:
            # Optionally add a learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        """
        for batch_idx, (features, labels, batch_weights) in enumerate(trainloader):
            features = features.float()
            labels = labels.view(-1, 1)
            batch_weights = batch_weights.float()
            
            if lbfgs:
                # Define closure for LBFGS optimizer
                def closure():
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = loss_fn(outputs, labels, batch_weights)
                    loss.backward()
                    return loss
                
                # Perform optimization step
                optimizer.step(closure)
                
                # Optionally retrieve the loss value explicitly
                loss = closure()
            else:
                # For standard optimizers
                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels, batch_weights)
                loss.backward()
                optimizer.step()
    
            # Debug logging
            if debug and batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.5f}")
        """       
        if not lbfgs:
            # Step the scheduler after each epoch
            scheduler.step()
        """
    return model

def weighted_mse_loss(inputs, target, weights=None):
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    if isinstance(inputs,np.ndarray):
        inputs=torch.from_numpy(inputs)
    if weights==None:
        weights = torch.ones_like(inputs)

    # Compute weighted MSE
    weighted_diff = weights * (inputs - target) ** 2
    return 0.5 * torch.sum(weighted_diff) #  / torch.sum(weights)

def weighted_kl_loss(inputs, target, weights=None):
    """
    Compute a weighted loss similar to KL Divergence.

    Parameters:
        inputs (torch.Tensor or np.ndarray): Predicted logits or probabilities.
        target (torch.Tensor or np.ndarray): Target probabilities.
        weights (torch.Tensor or None): Weights for each sample. If None, uniform weights are used.

    Returns:
        torch.Tensor: The computed weighted KL divergence loss.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).float()
    
    # Ensure inputs and target are in the range [0, 1]
    inputs_01 = torch.clip(0.5 * (inputs + 1), 1e-5, 1 - 1e-5)  # Avoid log(0)
    target_01 = torch.clip(0.5 * (target + 1), 1e-5, 1 - 1e-5)  # Avoid log(0)
    
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(inputs_01)

    # Compute the KL divergence term
    kl_div = target_01 * torch.log(target_01 / inputs_01) + (1 - target_01) * torch.log((1 - target_01) / (1 - inputs_01))
    
    # Apply weights and sum the loss
    weighted_kl = weights * kl_div
    return torch.sum(weighted_kl)
    
def weighted_bce_loss(inputs, target, weights=None):
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).double()  # Convert to double (float64)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).double()  # Convert to double (float64)

    # Ensure inputs and targets are within [0, 1] for BCE loss
    inputs_01 = torch.clamp(0.5 * (inputs + 1), 0, 1).double()  # Ensure double type
    targets_01 = torch.clamp(0.5 * (target + 1), 0, 1).double()  # Ensure double type

    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(inputs_01).double()  # Ensure weights are double
    elif isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float64)  # Ensure weights are double
        
    # Ensure weights have the shape [batch_size, 1]
    if weights.ndimension() == 1:
        weights = weights.unsqueeze(1)  # Add an extra dimension to make shape [batch_size, 1]
    
    # Compute the BCE loss
    loss_bce = nn.BCELoss(weight=weights)
    return loss_bce(inputs_01, targets_01)

def weighted_bce_logit_loss(inputs, target, weights=None):
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).double()  # Convert to double (float64)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs).double()  # Convert to double (float64)
        
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(inputs).double()  # Ensure weights are double
    elif isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float64)  # Ensure weights are double

    # Ensure weights have the shape [batch_size, 1]
    if weights.ndimension() == 1:
        weights = weights.unsqueeze(1)  # Add an extra dimension to make shape [batch_size, 1]
    
    # Compute the BCEWithLogits loss
    loss_bce_logit = nn.BCEWithLogitsLoss(weight=weights)
    return loss_bce_logit(inputs, target)

def f1_loss(predict, target, weights=None):
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).double()  # Convert to double (float64)
    target = 0.5 * (target + 1)
    if isinstance(predict, np.ndarray):
        predict = torch.from_numpy(predict).double()  # Convert to double (float64)
    predict = torch.clamp(0.5 * (predict + 1), 0, 1)  # Ensure values are within [0, 1
    
    # If weights are not provided, use uniform weights
    if weights is None:
        weights = torch.ones_like(target).double()  # Ensure weights are double
    elif isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float64)  # Ensure weights are double

    # Ensure weights have the shape [batch_size, 1]
    if weights.ndimension() == 1:
        weights = weights.unsqueeze(1)  # Add an extra dimension to make shape [batch_size, 1]]

    # Initialize BCEWithLogitsLoss once, passing weight
    bce_loss_fn = nn.BCEWithLogitsLoss(weight=weights)

    loss = 0
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += bce_loss_fn(predict[:, lack_cls], target[:, lack_cls])

    tp = predict * target
    tp = tp.sum(dim=0)
    
    fp = predict * (1 - target)
    fp = fp.sum(dim=0)
    
    fn = ((1 - predict) * target)
    fn = fn.sum(dim=0)
    
    tn = (1 - predict) * (1 - target)
    tn = tn.sum(dim=0)
    
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-8)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-8)
    cost_class1 = 1 - soft_f1_class1  # Reduce 1 - soft_f1_class1 to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # Reduce 1 - soft_f1_class0 to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # Take into account both class 1 and class 0
    macro_cost = cost.mean()  # Average on all labels
    
    return macro_cost + loss


# Base class to handle common initialization logic
class BaseAsymmetricMLP(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, beta, activation_fn=torch.tanh):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.activation_fn = activation_fn  # Accept activation function as a parameter
        self.init_weights()

    def init_weights(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(weight_init)

class AsymmetricMLP(BaseAsymmetricMLP):
    def __init__(self, input_size, hidden_size, alpha, beta, dropout_prob=0.0, activation_fn=torch.tanh, output_act=1):
        super().__init__(input_size, hidden_size, alpha, beta, activation_fn)
        self.hidden0 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout
        self.out = nn.Linear(hidden_size, 1)  # Hidden to output layer

        # Ensure double precision
        self.hidden0 = self.hidden0.to(torch.float64)
        self.out = self.out.to(torch.float64)

        # Mode determines the output activation function
        self.output_act = output_act

    def forward(self, x):
        x = x.to(torch.float64)  # Ensure input is float64
        o = self.activation_fn(self.hidden0(x))  # Apply activation to hidden layer
        o = self.dropout(o)  # Apply dropout
        z = self.out(o)  # Output layer

        # Apply different output activation functions based on output_mode
        if self.output_act == 1:
            return torch.where(
                z < 0,
                torch.tanh(z) * (1 - 2 * self.alpha),
                torch.tanh(z) * (1 - 2 * self.beta),
            )
        elif self.output_act == 2:
            return torch.where(
                z < 0,
                torch.tanh(z / (1 - 2 * self.alpha)) * (1 - 2 * self.alpha),
                torch.tanh(z / (1 - 2 * self.beta)) * (1 - 2 * self.beta),
            )
        else:
            raise ValueError("Invalid output_mode. Choose 1 or 2.")

# Mapping for activation functions
ACTIVATION_FUNCTIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "none": lambda x: x  # Identity function if no activation is needed
}

# Mapping for loss functions
LOSS_FUNCTIONS = {
    "MSE": weighted_mse_loss,
    "KL": weighted_kl_loss,
    "BCE": weighted_bce_loss,
    "BCE_logit": weighted_bce_logit_loss,
    "F1": f1_loss,
}

class LSEnsemble(nn.Module):
    def __init__(self, hidden_size, num_experts, alpha=0, beta=0, QC=1, Q_RB_C=1, 
                 Q_RB_S=1, n_epoch=1, n_batch=1, mode='random', 
                 input_size=None, drop_out=0, base_learner='FAMLP', 
                 activation_fn="tanh", output_act=1, optim='lbfgs', loss_fn='MSE'):  
        """
        Parameters:
        - hidden_size (int): 
            Number of neurons in the hidden layer for each expert in the ensemble.
        - num_experts (int): 
            Number of experts (individual models) in the ensemble.
        - alpha (float, optional, default=0): 
            Switching factor: majority to minority.
        - beta (float, optional, default=0): 
            Switching factor: minority to majority.
        - Q_RB_C (float, optional, default=1): 
            Classification cost.
        - Q_RB_S (float, optional, default=1): 
            Rebalancing factor for neutral population rebalance (SMOTE).
        - n_epoch (int, optional, default=1): 
            Number of training epochs for the ensemble.
        - n_batch (int, optional, default=1): 
            Batch size used during training.
        - mode (str, optional, default='random'): 
            Mode of sampling or initialization for the ensemble. Options may include:
            'random', 'class_equitative', or others depending on the implementation.
        - input_size (int, optional, default=None): 
            Number of input features in the dataset. This must be set before initializing the model.
        - drop_out (float, optional, default=0): 
            Dropout probability for the hidden layer to prevent overfitting.
        - base_learner (str, optional, default='FAMLP'):
            Type of base learner used in the ensemble:
                - 'FAMLP': Full Asymmetric MLP.
                - 'LogReg': Customized Logistic Regression.
                - 'Parzen': MLP Bayes.
                - 'AMLP': Simple Asymmetric MLP.
        - activation_fn (str, optional, default="tanh"): 
            Activation function for the hidden layers of the model. Common options:
            'tanh', 'relu', or 'sigmoid'.
        - output_act (int, optional, default=1): 
            Type of activation function applied at the output layer. Options:
            - 1: Customized tanh-based activation function.
            - 2: Scaled tanh activation function.
        - optim (str, optional, default='lbfgs'): 
            Optimization method.
        - loss_fn (str, optional, default='MSE'): 
            Loss function used for training. Common options:
            - 'MSE': Mean Squared Error.
            - 'KL': Kullback-Leibler Divergence.
            - 'BCE': Binary Cross-Entropy for classification tasks.
            - 'F1': F1 Score
        """
        super(LSEnsemble, self).__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self.drop_out = drop_out
      
        self.Q_RB_C = Q_RB_C
        self.Q_RB_S = Q_RB_S
        
        if QC == None:
            C10 = 1  # False Positive (FP)
            C00 = 0  # True Negative (TN)
            C01 = 1  # False Negative (FN)
            C11 = 0  # True Positive (TP)
            self.QC = float(C10 - C00) / float(C01 - C11)
        else:
            self.QC = QC 
        
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.mode = mode
        self.base_learner = base_learner
        self.hidden_size = hidden_size
        # Convert string activation function to actual function
        if activation_fn not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Invalid activation function '{activation_fn}'. Choose from {list(ACTIVATION_FUNCTIONS.keys())}.")
        self.activation_fn = ACTIVATION_FUNCTIONS[activation_fn]

        self.optim = optim
        # Map the provided string to the actual loss function
        self.loss_fn_e = LOSS_FUNCTIONS.get(loss_fn, weighted_mse_loss)  # Default to MSE if not found

        self.output_mode = output_act

        # Initialize the experts if input_size is provided during initialization
        if input_size is not None:
            self.initialize_experts(input_size)

    def initialize_experts(self, input_size):
        if self.base_learner == 'AMLP':  # Use MLPClassifierTorch if lbfgs is set to 'MLP'
            self.experts = nn.ModuleList([
                MLPClassifierTorch(
                    input_dim=input_size,
                    hidden_layer_sizes=(self.hidden_size,),
                    activation=self.activation_fn.__name__,
                    alpha=self.alpha,   # Pass alpha
                    beta=self.beta,     # Pass beta
                    batch_size=self.n_batch,
                    max_iter=self.n_epoch,
                    solver='adam', # 'lbfgs'#  if self.lbfgs else 'adam'
                ) for _ in range(self.num_experts)
            ])
        elif self.base_learner == 'LogReg':
            self.experts = nn.ModuleList([
                LogisticRegressionTorch(
                    input_dim=input_size,
                    alpha=self.alpha,
                    beta=self.beta,
                    num_epochs=self.n_epoch
                ) for _ in range(self.num_experts)
            ])
        else:
            # Create experts with the actual input size
            self.experts = nn.ModuleList([
                AsymmetricMLP(
                    input_size,
                    self.hidden_size,
                    self.alpha,
                    self.beta,
                    self.drop_out,
                    self.activation_fn,
                    self.output_mode
                ) for _ in range(self.num_experts)
            ])
            
        

    def generate_experts_data(self, x, y, w=None, Q_RB_S=1, RB_each_expert=True):
        """
        Generates rebalanced and augmented datasets for experts using SMOTE, while adjusting weights and applying label switching.
    
        Parameters:
        -----------
        x : torch.Tensor
            Features tensor (N_samples, N_features).
        y : torch.Tensor
            Labels tensor (N_samples,).
        w : torch.Tensor, optional
            Sample weights tensor (N_samples,). Defaults to uniform weights if not provided.
        Q_RB_S : float, optional
            Desired ratio of the majority to minority class in the rebalanced dataset. Defaults to 1 (no rebalancing).
        RB_each_expert : bool, optional
            If True, applies SMOTE separately for each expert. If False, shares the same rebalanced dataset among all experts.
    
        Workflow:
        ---------
        1. Converts input tensors `x`, `y`, and `w` to numpy arrays for SMOTE compatibility.
        2. Handles binary labels (supports both 0/1 and -1/+1 formats) and computes class proportions.
        3. Applies SMOTE to rebalance the dataset, optionally extending sample weights for synthetic data.
        4. Adjusts labels and weights for each expert based on label switching parameters `alpha` and `beta`.
        5. Converts the final datasets back to PyTorch tensors and assigns them to each expert.
    
        Notes:
        ------
        - SMOTE generates synthetic samples for the minority class to achieve the desired class ratio.
        - If `RB_each_expert` is True, each expert gets a unique rebalanced dataset; otherwise, all share the same.
        - Label switching adjusts targets (`y`) and weights (`w`) based on predefined perturbation factors (`alpha` and `beta`).
    
        """
        x_np, y_np = x.cpu().numpy(), y.cpu().numpy()
        w_np = w.cpu().numpy() if w is not None else np.ones_like(y_np, dtype=np.float32)
    
        # Binary label handling
        unique_labels = np.unique(y_np)
        
        if set(unique_labels) == {0, 1}:  # Case for binary {0, 1}
            self.bin_format = 0
            y_np = np.where(y_np == 0, -1, 1)
        elif set(unique_labels) == {-1, 1}:  # Case for binary {-1, 1}
            self.bin_format = -1
        
        # Compute the class counts for -1 and 1
        N0_tr = np.sum(y_np == -1)
        N1_tr = np.sum(y_np == 1)
        
        # Prevent division by zero if classes are highly imbalanced
        if N1_tr == 0:
            QP_tr = 1000  # Or some large value to indicate extreme imbalance
        else:
            P0_tr = N0_tr / (N0_tr + N1_tr)
            P1_tr = N1_tr / (N0_tr + N1_tr)
            QP_tr = P0_tr / P1_tr if P1_tr >= eps else 1000
        
        self.QP_tr = QP_tr
    
        Q_RB_S = max(Q_RB_S, 1)
        sampling_strategy = min(Q_RB_S / QP_tr, 1.0)
    
        def apply_smote(x_np, y_np, w_np):
            """
            Applies SMOTE to rebalance the dataset and adjusts weights for synthetic samples.
            
            Returns:
            --------
            X_RB : numpy.ndarray
                Features of the rebalanced dataset.
            y_RB : numpy.ndarray
                Labels of the rebalanced dataset.
            w_RB : numpy.ndarray
                Weights of the rebalanced dataset.
            """
            try:
                smote = SMOTE(random_state=random.randint(1, 100), sampling_strategy=sampling_strategy)
                X_RB, y_RB = smote.fit_resample(x_np, y_np)
                n_synthetic = len(X_RB) - len(x_np)
                if n_synthetic > 0:
                    minority_class = min(y_np)
                    avg_weight = np.mean(w_np[y_np == minority_class])
                    synthetic_weights = np.full(n_synthetic, avg_weight)
                    w_RB = np.concatenate([w_np, synthetic_weights])
                else:
                    w_RB = w_np
                return X_RB, y_RB, w_RB
            except ValueError:
                return x_np, y_np, w_np
    
        def process_labels_and_weights(X_RB, y_RB, w_RB):
            """
            Processes rebalanced labels and weights, including optional label switching.
            
            Returns:
            --------
            X_RB : torch.Tensor
                Features tensor for the expert.
            y_RB_SW : torch.Tensor
                Adjusted labels tensor for the expert.
            w_RB_SW : torch.Tensor
                Adjusted weights tensor for the expert.
            """
            targets_sw, w_RB_SW = (y_RB, w_RB) if self.alpha == 0 and self.beta == 0 else label_switching(y_RB, w_RB, self.alpha, self.beta)
            targets_sw = torch.from_numpy(targets_sw).to(x.device)
            w_RB_SW = torch.from_numpy(w_RB_SW).to(x.device)
            y_RB_SW = torch.where(targets_sw > 0, (1 - 2 * self.beta), -(1 - 2 * self.alpha))
            return torch.from_numpy(X_RB).float().to(x.device), y_RB_SW, w_RB_SW
    
        if Q_RB_S > 1:
            # self.QP_RB_tr = QP_tr/Q_RB_S # Commented (Verified) 
            if RB_each_expert:
                # Generate unique SMOTE data for each expert
                for expert in self.experts:
                    X_RB, y_RB, w_RB = apply_smote(x_np, y_np, w_np)
                    expert.X, expert.y, expert.w = process_labels_and_weights(X_RB, y_RB, w_RB)
            else:
                # Generate a single rebalanced dataset shared by all experts
                X_RB, y_RB, w_RB = apply_smote(x_np, y_np, w_np)
                for expert in self.experts:
                    expert.X, expert.y, expert.w = process_labels_and_weights(X_RB, y_RB, w_RB)
        else:
            # No rebalancing; all experts use the original dataset
            for expert in self.experts:
                expert.X, expert.y, expert.w = process_labels_and_weights(x_np, y_np, w_np)

    def fit(self, x_train, y_train, sample_weight=None):
        """
        Fit the ensemble model using training data.
    
        Parameters:
        - x_train: Tensor or NumPy array of shape (n_samples, n_features) for training data.
        - y_train: Tensor or NumPy array of shape (n_samples,) for training labels.
        - sample_weight: Optional tensor or NumPy array of shape (n_samples,) for sample weights.
        """
        # Ensure x_train and y_train are PyTorch tensors
        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()  # Use float to handle potential binary labels
    
        # Handle sample_weight
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray):
                sample_weight = torch.from_numpy(sample_weight).float()
        else:
            # Initialize weights to 1.0 for all samples if no weights are provided
            sample_weight = torch.ones_like(y_train, dtype=torch.float32)
            
        # Ensure the data is on the same device as the model
        # Check if model parameters exist and get the device, otherwise fallback to CPU
        if len(list(self.parameters())) > 0:
            device = next(self.parameters()).device
        else:
            device = torch.device("cpu")  # Fallback to CPU if no model parameters exist

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        sample_weight = sample_weight.to(device)
        
        self.generate_experts_data(x_train, y_train, sample_weight,
                                   Q_RB_S=self.Q_RB_S,
                                   RB_each_expert=False)
        
        # Optional: Set sample weights to experts if provided
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray):
                sample_weight = torch.from_numpy(sample_weight).float()
            sample_weight = sample_weight.to(device)
            self.fit_expert_model(sample_weight, epochs=self.n_epoch, batch_size=self.n_batch, optim=self.optim)
        else:
            # If no sample weights are provided, call fit_expert_model with default weights
            default_weights = torch.ones(y_train.shape[0]).float().to(device)
            self.fit_expert_model(default_weights, epochs=self.n_epoch, batch_size=self.n_batch, optim=self.optim)
        
        return self
                
    def fit_expert_model(self, w_train, epochs=50, batch_size=256, optim='lbfgs'):
        """
        Train experts using their stored data with either LBFGS or RMSprop optimization.
    
        Parameters:
        - w_train: Training weights (tensor).
        - epochs: Number of training epochs (int).
        - batch_size: Batch size for training (int).
        - lbfgs: Whether to use the LBFGS optimizer (bool).
    
        Returns:
        - self: The updated model after training.
        """
        # Initialize weights for each expert
        weights = torch.ones((self.experts[0].y.shape[0], self.num_experts)).to(w_train.device)
    
        # Compute weights for each expert
        for i, expert in enumerate(self.experts):
            # weights[:, i] = compute_weights(expert.y, RB=self.Q_RB_C, IR=1, mode='Normal') * w_train  # Scale by training weights
            weights[:, i] = compute_weights(expert.y, RB=self.Q_RB_C, IR=1, mode='Normal') * expert.w  # Scale by training weights
    
        # Training loop for experts
        if optim == 'lbfgs':
            # Use LBFGS optimizer for each expert
            for i, expert in enumerate(self.experts):
                # Configure the LBFGS optimizer
                optim_LBFGS_scipy = LBFGSScipy(
                    expert.parameters(),
                    max_iter=150,
                    max_eval=150,
                    tolerance_grad=1e-04,
                    tolerance_change=10e6 * eps,  # `eps` is assumed to be defined outside
                    history_size=10
                )
    
                # Train the model using LBFGS and the integrated data loader functionality
                train_model(
                    expert.X,  # Input data for this expert
                    expert.y,  # Labels for this expert
                    expert,  # Model to train (expert)
                    self.loss_fn_e,  # Loss function
                    optim_LBFGS_scipy,  # Optimizer (LBFGS)
                    weights[:, i],  # Sample weights for this expert
                    num_epochs=epochs,  # Number of epochs
                    batch_size=None, # batch_size,  # Batch size
                    mode=self.mode, #'representative', # 'class_equitative',  # Mode for DataLoader
                    lbfgs = True, 
                    debug=False # True  # Enable debug information if necessary
                )
        elif (self.base_learner == 'Parzen') or (self.base_learner == 'AMLP'):
            for i, expert in enumerate(self.experts):
                x_np, y_np = expert.X.cpu().numpy(), expert.y.cpu().numpy()
                expert.fit(x_np, y_np)
        else:
            # Main training loop with gradient-based optimizers (e.g., RMSprop)
            for i, expert in enumerate(self.experts):
                # Select the optimizer based on self.optim
                if self.optim == 'adam':
                    optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)
                elif self.optim == 'adamw':
                    optimizer = torch.optim.AdamW(expert.parameters(), lr=0.01, weight_decay=1e-2)
                elif self.optim == 'rmsprop':
                    optimizer = torch.optim.RMSprop(expert.parameters(), lr=0.001)
                elif self.optim == 'sgd':
                    optimizer = torch.optim.SGD(expert.parameters(), lr=0.1, momentum=0.9)
                elif self.optim == 'adagrad':
                    optimizer = torch.optim.Adagrad(expert.parameters(), lr=0.01)
                elif self.optim == 'adadelta':
                    optimizer = torch.optim.Adadelta(expert.parameters(), lr=0.01)
                else:
                    raise ValueError(f"Unsupported optimizer: {self.optim}")
                
                 # Train the expert using the selected optimizer
                train_model(
                    expert.X,  # Input data for this expert
                    expert.y,  # Labels for this expert
                    expert,  # Model to train (expert)
                    self.loss_fn_e,  # Loss function
                    optimizer,  # Selected optimizer
                    weights[:, i],  # Sample weights for this expert
                    num_epochs=epochs,  # Number of epochs
                    batch_size=batch_size,  # Batch size
                    mode=self.mode,  # Training mode
                    lbfgs=(self.optim == 'lbfgs'),  # Use LBFGS-specific behavior
                    debug=False  # Enable debug if necessary
                )
    
        return self
    
    # Get outputs from all experts
    def get_expert_outputs(self, x):
        expert_outputs = self.experts[0](x)
        aux_outputs = torch.zeros_like(expert_outputs)
        for i in range(self.num_experts-1):
            aux_outputs = self.experts[i+1](x)
            expert_outputs = torch.cat((expert_outputs, aux_outputs), 1)
        return expert_outputs

    # Predict outputs of each expert
    def predict_expert_outputs(self, x):
        with torch.no_grad():
            return self.get_expert_outputs(x) # .double())

    def forward(self, x):
        """
        This method performs the forward pass by calculating the predictions from the experts
        and returning the averaged prediction (o_pred) across experts.
        """
    
        # Convert input to tensor
        x_torch = torch.from_numpy(x).float()
    
        # Get expert outputs and compute their average
        expert_outputs = self.predict_expert_outputs(x_torch)
        o_pred = expert_outputs.mean(dim=1)  # Average over experts
        
        return o_pred.numpy()  # Forward method now gives output in the 0 to 1 range
    
    
    def predict(self, x):
        """
        Final prediction averaging over experts and applying threshold to obtain class labels.
        """
        QP_tr = self.QP_tr
        Q_tr = self.QC * self.QP_tr
        
        Q_RB_C = self.Q_RB_C
        Q_RB_S = self.Q_RB_S
        
        # Q_tr = self.QC * self.QP_tr # Commented (Verified) 
        
        QR_tr_expr = 1
        if QR_tr_expr == 1:
            # QR_tr is the inverse of the product of Q_RB_C and Q_RB_S
            QR_tr = QP_tr / (Q_RB_C * Q_RB_S)
        elif QR_tr_expr == 2: # Best case
            if (Q_RB_C == 1) and (Q_RB_S == 1):
                QR_tr = 1*QP_tr 
            else:
                # QR_tr is the average of the inverses of Q_RB_C and Q_RB_S
                QR_tr = 1*QP_tr * ((Q_RB_S + Q_RB_C) / (Q_RB_C * Q_RB_S))
        elif QR_tr_expr == 3:
            # QR_tr is half the average of the inverses of Q_RB_C and Q_RB_S
            QR_tr = 0.5 * QP_tr * ((Q_RB_S + Q_RB_C) / (Q_RB_C * Q_RB_S))
        elif QR_tr_expr == 4: # Second Place. Bacc = 0.85672
            # QR_tr is the geometric average of the inverses of Q_RB_C and Q_RB_S
            QR_tr = QP_tr / np.sqrt(Q_RB_C * Q_RB_S)
        elif QR_tr_expr == 5: # Reciprocal Sum (Harmonic Mean variant)
            QR_tr = QP_tr * np.sqrt(2 / (1/Q_RB_C + 1/Q_RB_C))
               
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
    
        # Apply thresholding to get the final class labels
        y = np.ones_like(o_pred)
        eta_th = (2 * (self.alpha + (1 - self.alpha - self.beta) * (Q_tr / (Q_tr + QR_tr))) - 1)
        y[o_pred < eta_th] = self.bin_format
        
        return y.astype(int)
    
    def predict_proba(self, x):
        """
        Returns the predicted probability (o_pred) without applying threshold logic.
        This method is equivalent to predict but returns o_pred instead of y.
        """
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
        
        # Normalize to be between 0 and 1 (for proba outputs)
        o_pred_01 = (o_pred + 1) / 2
        
        return o_pred_01.numpy().astype(int)  # Convert back to numpy if necessary

    # Method for GridSearchCV compatibility
    def get_params(self, deep=True):
        return {
            'hidden_size': self.hidden_size,
            'num_experts': self.num_experts,
            'alpha': self.alpha,
            'beta': self.beta,
            'Q_RB_C': self.Q_RB_C,
            'Q_RB_S': self.Q_RB_S,
            'n_epoch': self.n_epoch
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

from sklearn.base import BaseEstimator
    
class LSEnsembleWrapper(BaseEstimator):
    def __init__(self,  hidden_size, num_experts, alpha=0, beta=0, Q_RB_C=1, 
                 Q_RB_S=1, n_epoch=1, n_batch=1, lbfgs=False, mode='random', 
                 input_size=None, drop_out=0, activation_fn="tanh", output_act=1,
                 loss_fn='MSE'):  # Added loss_fn as an argument):
        """
        Wrapper for the LSEnsemble model to allow use in scikit-learn-style interfaces.

        Parameters:
        - hidden_size (int): 
            Number of neurons in the hidden layer for each expert in the ensemble.
        - num_experts (int): 
            Number of experts (individual models) in the ensemble.
        - alpha (float, optional, default=0): 
            Switching Factor: majority to minority
        - beta (float, optional, default=0): 
            Switching Factor: minority to majority
        - Q_RB_C (float, optional, default=1): 
            Classification cost
        - Q_RB_S (float, optional, default=1): 
            Rebalancing factor for neutral population rebalance (SMOTE)
        - n_epoch (int, optional, default=1): 
            Number of training epochs for the ensemble.
        - n_batch (int, optional, default=1): 
            Batch size used during training.
        - lbfgs (bool, optional, default=False): 
            Whether to use the LBFGS optimizer instead of other optimization algorithms.
        - mode (str, optional, default='random'): 
            Mode of sampling or initialization for the ensemble. Options may include:
            'random', 'class_equitative', or others depending on the implementation.
        - input_size (int, optional, default=None): 
            Number of input features in the dataset. This must be set before initializing the model.
        - drop_out (float, optional, default=0): 
            Dropout probability for the hidden layer to prevent overfitting.
        - activation_fn (str, optional, default="tanh"): 
            Activation function for the hidden layers of the model. Common options:
            'tanh', 'relu', or 'sigmoid'.
        - output_act (int, optional, default=1): 
            Type of activation function applied at the output layer. Options:
            - 1: Customized tanh-based activation function.
            - 2: Scaled tanh activation function.
        - loss_fn (str, optional, default='MSE'): 
            Loss function used for training. Common options:
            - 'MSE': Mean Squared Error.
            - 'KL': Kull-back Leibler Divergence
            - 'BCE': Binary Cross Entropy for classification tasks.
        
        """
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self.Q_RB_C = Q_RB_C
        self.Q_RB_S = Q_RB_S
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.lbfgs = lbfgs
        self.mode = mode
        self.input_size = input_size
        self.drop_out = drop_out
        self.activation_fn = activation_fn
        self.output_act = output_act
        self.loss_fn = loss_fn
        
        self.model = LSEnsemble(hidden_size, num_experts, alpha, beta, Q_RB_C, 
                                Q_RB_S, n_epoch, n_batch, lbfgs, mode, 
                                input_size, drop_out, activation_fn, 
                                output_act, loss_fn)

    def fit(self, X, y):
        """
        Fit the LSEnsemble model using training data.
        
        Parameters:
        - X: Training features (numpy array or torch tensor).
        - y: Training labels (numpy array or torch tensor).
        
        Returns:
        - self: The fitted model.
        """
        # Call the fit method of the LSEnsemble model with proper arguments
        self.model.fit(X, y)  # Remove `self` from the arguments
        
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.model.predict(X) 