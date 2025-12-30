import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import WeightedRandomSampler

from transformers import Trainer

from typing import Optional

from sklearn.utils.class_weight import compute_class_weight


class WeightedTrainer(Trainer):

    def __init__(
            self, 
            *args, 
            class_weights: Optional[torch.Tensor] = None, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.args.device)
        
    def _get_train_sampler(self):

        labels = torch.LongTensor(self.train_dataset['label'])
        class_sample_counts = torch.bincount(labels)
        class_weights_sampling = 1.0 / class_sample_counts.float()
        sample_weights = class_weights_sampling[labels.long()]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Sample with replacement to allow oversampling minority classes
        )
        return sampler

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")

        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels.device != logits.device:
            labels = labels.to(logits.device)

        if self.class_weights is not None:
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            
            weights = torch.where(
                labels==1, 
                self.class_weights[1], 
                self.class_weights[0]
                )
            # Use CrossEntropyLoss with class weights
            # loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            # loss = loss_fct(logits, labels.view(-1,1))
            loss = binary_cross_entropy_with_logits(
                input=logits.flatten(), 
                target=torch.Tensor(labels),
                weight=weights,
                )
        else:
            # Standard loss computation (same as default Trainer)
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits, labels.view(-1,1))
            loss = binary_cross_entropy_with_logits(
                input=logits.flatten(), 
                target=torch.Tensor(labels),
                )
        
        return (loss, outputs) if return_outputs else loss


def calculate_balance_weight(labels):

    num_classes = np.unique(labels).shape[0]
    class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=labels
        )
    
    return torch.tensor(class_weights, dtype=torch.float32)