import torch
import copy

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def print_exp_details(args):
    print('\nExperimental details:')
    print(f"    Model     : {args['model']}")
    print(f"    Optimizer : {args['optimizer']}")
    print(f"    Learning  : {args['lr']}")
    print(f"    Global Rounds   : {args['epochs']}\n")

    print('    Federated parameters:')
    if args["iid"]:
        print('    IID')
    else:
        print('    Non-IID')
    print(f"    Fraction of users  : {args['frac']}")
    print(f"    Local Batch size   : {args['local_bs']}")
    print(f"    Local Epochs       : {args['local_ep']}\n")

def print_clientwise_class_distribution(dataset, classes, users_or_classes, group):
    def create_labels(classes):
        labels = dict()
        for i in range(classes):
            labels[i] = 0
        return labels

    for j in range(users_or_classes):
        labels = create_labels(classes)
        for i in group[j]:
            labels[dataset[int(i)][1]] += 1
        print(f"Data distribution for client : {j} :::: { labels}")


def get_group_contribution(contributions):
    """
    Get the average contribution of the group.
    """
    avg_contributions = dict()
    for key in contributions[0].keys():
        avg_contributions[key] = torch.zeros_like(contributions[0][key])
        for i in range(len(contributions)):
            avg_contributions[key] += contributions[i][key]
        avg_contributions[key] = torch.div(avg_contributions[key], len(contributions))
    return avg_contributions

def get_avg_client_contributions(client_contributions):
        """
        Get the average contribution of the clients.
        Args:
            client_contributions: dict[client_id] = [contribution1, contribution2, ...]
        Returns:
            avg_contributions: dict[client_id] = average_contribution
        """
        avg_contributions = dict()
        for client_id, contributions in client_contributions.items():
            avg_contribution = dict()
            for param in contributions[0].keys():
                avg_contribution[param] = torch.zeros_like(contributions[0][param])
                for contribution in contributions:
                    avg_contribution[param] += contribution[param]
                avg_contribution[param] /= len(contributions)
            avg_contributions[client_id] = avg_contribution
        return avg_contributions


def get_client_wise_differences(global_model, new_local_weights):
    """
    Get the client wise differences.
    Args:
        global_model: The global model.
        new_local_weights: The new local weights.
    Returns:
        client_wise_differences: dict[client_id] = difference
    """
    client_wise_differences = dict()
    for client_id, local_weight in enumerate(new_local_weights):
        difference = dict()
        for param in global_model.keys():
            difference[param] = torch.abs(global_model[param] - local_weight[param])
        client_wise_differences[client_id] = difference
    return client_wise_differences



