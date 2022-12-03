import torch
import torch.nn.functional as F


class EntropyMaximization(torch.nn.Module):
    """Entropy Maximization loss

    Arguments:
        t : temperature
    """

    def __init__(self, t=1.):
        super(EntropyMaximization, self).__init__()
        self.t = t

    def forward(self, lbl, pred):
        """Compute loss.

        Arguments:
            lbl (torch.tensor:float): predictions, not confidence, not label.
            pred (torch.tensor:float): predictions.

        Returns:
            loss (torch.tensor:float): entropy maximization loss

        """
        loss = torch.mean(torch.sum(F.softmax(lbl/self.t, dim=-1) * F.log_softmax(pred/self.t, dim=-1), dim=-1))
        return loss
