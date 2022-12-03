import numpy as np
import torch

from __main__ import args


def compute_accuracy_class(data_iter, net, device):
    """Compute the classification accuracy

    Arguments:
        data_iter : data iterator
        net : CNN for predicting
        device : cpu or gpu

    Returns:
        cla_acc : accuracy for each class
        all_acc : accuracy for all

    """
    # net evaluation mode
    net.eval()

    sum_metric = np.zeros((args.num_classes,), dtype=np.int64)
    num_instance = np.zeros((args.num_classes,), dtype=np.int64)

    with torch.no_grad():
        for batch in data_iter:
            data = batch['image']
            lbl = batch['label']
            data = data.to(device)
            y_pred = net(data)

            y_pred = torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
            lbl = lbl.numpy()
            pred_true = y_pred == lbl

            for i, cla in enumerate(lbl):
                sum_metric[cla] += pred_true[i]
                num_instance[cla] += 1

    cla_acc = sum_metric/num_instance
    all_acc = np.sum(sum_metric)/np.sum(num_instance)

    return cla_acc, all_acc
