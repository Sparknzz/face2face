import torch.nn as nn

class Face2Face_loss(nn.Module):
    def __init__(self, gamma = 1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.gamma = gamma

    def forward(self, features, labels):
        f_logits, l_logits, r_logits, t_logits, fusion_logits = features

        l1 = self.criterion(f_logits, labels)
        l2 = self.criterion(l_logits, labels)
        l3 = self.criterion(r_logits, labels)
        l4 = self.criterion(t_logits, labels)
        l_fusion = self.criterion(fusion_logits, labels)
        
        l_total = l1 + 0.5 * l2 + 0.5 * l3 + l4 + l_fusion

        return l_total