import torch
import random
import os
import sys
import logging
import numpy as np
from shutil import copy
from datetime import datetime
import collections
from sklearn.preprocessing import StandardScaler
from torch.optim.optimizer import Optimizer, required
from sklearn.metrics import classification_report, accuracy_score
def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    for filepath,dirnames,filenames in os.walk(r'src/seismicface'):
        # print(filepath, dirnames, filenames)
        for filename in filenames:
            if 'py' in filename:
                copy(os.path.join(filepath, filename), os.path.join(destination_dir, filename))
class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=0.001,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")
    
def save_checkpoint(home_path, model, name, hparams, epoch = 0):
    save_dict = {
        "dataset": name,
        "hparams": dict(hparams),
        "fe": model[0].state_dict(),
        "te": model[1].state_dict(),
        "clf": model[2].state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, "checkpoint_{}.pt".format(epoch))

    torch.save(save_dict, save_path)
def get_mask(b):
    l = torch.tensor(np.sqrt(b)).int()
    m1 = np.eye(l*l,l*l)
    m2 = np.eye(l*l,l*l,k=-1)
    m3 = np.eye(l*l,l*l,k=+1)
    m4 = np.eye(l*l,l*l,k=-l)
    m5 = np.eye(l*l,l*l,k=-l-1)
    m6 = np.eye(l*l,l*l,k=-l+1)
    m7 = np.eye(l*l,l*l,k=l)
    m8 = np.eye(l*l,l*l,k=l-1)
    m9 = np.eye(l*l,l*l,k=l+1)
    m = m2+m3+m4+m5+m6+m7+m8+m9
    # np.sum(m,axis=-1)
    m[torch.arange(l)*l,:] = 0
    m[torch.arange(l)*l+l-1,:] = 0
    m[:l,:] = 0
    m[-l:,:] = 0
    m = torch.from_numpy(m)
    return m
def calc_metrics(pred_labels, true_labels):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)
    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy * 100, r["macro avg"]["f1-score"] * 100

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, ssl_method, encoder,paraname,train_mode, exp_log_dir):
    log_dir = os.path.join(exp_log_dir,'logger')
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Train Mode:  {train_mode}')
    logger.debug(f'Method:  {ssl_method}')
    logger.debug(f'Encoder:  {encoder}')
    logger.debug(f'Paraname:  {paraname}')
    logger.debug("=" * 45)
    return logger, log_dir

def xnorm(x,drop=False,upper=95,floor=5):
    # w = gauss_window(w, sigma)
    if drop:
        x = drop_error_value(x,upper=upper,floor=floor)
    scaler = StandardScaler()
    shape = x.shape
    x = scaler.fit_transform(x.reshape(-1,1))
    x = x.reshape(shape)
    return x
def gauss_window(w,sigma):
        # Create a tensor for the window
        window = np.arange(w)
        # Calculate the Gaussian function
        window = np.exp(-0.5 * ((window - (w) // 2) / sigma) ** 2)
        # Normalize the window
        # window /= window.sum()
        return window 
def drop_error_value(x,upper=95,floor=5):
    upper_value = np.percentile(x, upper)
    floor_value = np.percentile(x, floor)
    data = np.where(((x>upper_value)), upper_value, x)
    data = np.where(((x<floor_value)), floor_value, data)
    return data
    
def get_balance_class_oversample(x, y):
    """
    from deepsleepnet https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/utils.py
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y