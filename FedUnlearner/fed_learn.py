import torch

from FedUnlearner.utils import average_weights, get_avg_client_contributions
from copy import deepcopy

def fed_train(num_training_iterations, train_dataloader, test_dataloader, clientwise_dataloaders,
              global_model, num_local_epochs, device = 'cpu'):
    """
    """
    global_model.to(device)
    global_model.train()
    
    num_clients = len(clientwise_dataloaders)

    client_contributions = {}
    for client_idx in range(num_clients):
        client_contributions[client_idx] = []

    for iteration in range(num_training_iterations):
        print(f"Global Iteration: {iteration}")
        
        new_local_weights = []
        for client_idx in range(num_clients):
            print(f"Client: {client_idx}")
            client_dataloader = clientwise_dataloaders[client_idx]
            client_model = deepcopy(global_model)

            optimizer = None # create optimizer
            loss_fn = None # create loss function
            
            
            train_local_model(client_model = client_model, client_dataloader = client_dataloader, 
                              loss_fn = loss_fn, optimizer = optimizer, num_epochs = num_local_epochs, 
                              device = device)
            new_local_weights.append(client_model.state_dict())
            test_acc_client, test_loss_client = test_local_model(client_model, test_dataloader, loss_fn, device)
            print(f"Test Accuracy for client {client_idx} : {test_acc_client*100}, Loss : {test_loss_client}")
        

        client_wise_contributions = get_client_wise_differences(global_model, new_local_weights)
        for client_id, contribution in client_wise_contributions.items():
            client_contributions[client_id].append(contribution)

        # update gloal model
        updated_global_weights = average_weights(new_local_weights)
        global_model.load_state_dict(updated_global_weights)

        

        # evaluate global model
        test_acc_global, test_loss_global = test_local_model(global_model, test_dataloader, loss_fn, device)
        print(f"Test Accuracy for global model : {test_acc_global*100}, Loss : {test_loss_global}")
    
    # print client wise accuracy of the global model
    print_clientwise_accuracy(global_model, clientwise_dataloaders, device)

    # print class wise accuracy of the global model
    print_classwise_accuracy(global_model, test_dataloader, device)

    avg_client_contributions = get_avg_client_contributions(client_contributions)

    return global_model, avg_client_contributions

            


def train_local_model(model, dataloader, loss_fn, optimizer, num_epochs, device = 'cpu'):
    model = model.to(device)
    model.train()

    for iter in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            log_probs = model(images)
            loss = loss_fn(log_probs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def test_local_model(model, dataloader, loss_fn, device = 'cpu'):
    model = model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            test_loss += loss_fn(log_probs, labels).item()
            pred = log_probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    return test_acc, test_loss

def print_clientwise_accuracy(global_model, clientwise_dataloaders, device):
    """
    Print the clientwise accuracy.
    """
    for client_id, dataloader in clientwise_dataloaders.items():
        test_acc, _ = test_local_model(global_model, dataloader, device)
        print(f"Client {client_id} Test Accuracy : {test_acc*100}")

def print_classwise_accuracy(global_model, dataloader, device):
    """
    Print the classwise accuracy.
    """
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = global_model(images)
            _, predicted = torch.max(log_probs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f"Accuracy of {i} : {class_correct[i] / class_total[i]}")