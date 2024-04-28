import torch
import torchvision
import argparse

from FedUnlearner.utils import average_weights, print_exp_details, print_clientwise_class_distribution, \
    get_group_contribution, 
from FedUnlearner.data_utils import get_dataset, create_dirichlet_data_distribution, create_iid_data_distribution
from FedUnlearner.fed_learn import fed_train, test_local_model
from FedUnlearner.unlearn import apply_dampening

# create argument parser
parser = argparse.ArgumentParser(description='FedUnlearner')

# add arguments
parser.add_argument('--model', type=str, default='allcnn', options = ["allcnn"], help='model name')
parser.add_argument('--dataset', type=str, default='mnist', options = ["mnist","cifar10"], help='dataset name')
parser.add_argument('--optimizer', type=str, default='sgd', options = ["sgd"], help='optimizer name')
# provide indexes of clients which are to be forgotten, allow multiple clients to be forgotten
parser.add_argument('--forget_clients', type=int, nargs='+', default=[0], help='forget clients')
parser.add_argument('--apply_backdoor', type=bool, default=False, help='apply backdoor attack')
parser.add_argument('--total_num_clients', type=int, default=10, help='total number of clients')
parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
parser.add_argument('--client_data_distribution', type = str, default='dirichlet', 
                    options = ["dirichlet", "iid"], help='client data distribution')

if __name__ == "__main__":
    args = parser.parse_args()
    print_exp_details(args)

    # get the dataset
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset)

    # create client groups
    client_groups = None

    if args.client_data_distribution == 'dirichlet':
        client_groups = create_dirichlet_data_distribution(train_dataset, 
                                                           num_clients=args.total_num_clients, num_classes=num_classes)
    elif args.client_data_distribution == 'iid':
        client_groups = create_iid_data_distribution(train_dataset, num_clients=args.total_num_clients, 
                                                     num_classes=num_classes)
    else:
        raise "Invalid client data distribution"
    

    # print the clientwise class distribution
    print_clientwise_class_distribution(client_groups)


    # get the forget client

    if len(args.forget_clients) > 1:
        raise "Only one client forgetting supported at the moment."
    forget_client = args.forget_clients[0]
    
    # create dataloader for the forget client
    clientwise_dataloaders = None
    train_dataloader = None
    test_dataloader = None

    # train the model
    global_model = None
    retrained_global_model = None
    if args.model == 'allcnn':
        global_model = AllCNN(num_classes=num_classes)
    else:
        raise "Invalid model name"
    
    # train the model
    global_model, avg_client_contributions = fed_train(num_training_iterations = 10, train_dataloader = train_dataset, 
                                                       test_dataloader = test_dataset, 
                                                       clientwise_dataloaders = clientwise_dataloaders, 
                                                       global_model = global_model, num_local_epochs = 5, 
                                                       device = 'cpu')
    
    
    # evaluate attack accuracy
    if args.apply_backdoor:
        evaluate_backdoor_attack(global_model, backdoor_dataset,device = args.device)

    
    # train the model on retain data
    retain_clientwise_dataloaders = {key: value for key, value in clientwise_dataloaders.items() 
                                     if key not in args.forget_clients}
    
    retrained_global_model, avg_client_contributions = fed_train(num_training_iterations = 10, train_dataloader = train_dataset, 
                                                                test_dataloader = test_dataset, 
                                                                clientwise_dataloaders = retain_clientwise_dataloaders, 
                                                                global_model = global_model, num_local_epochs = 5, 
                                                                device = args.device)
    

    forget_client_contribution = [avg_client_contributions[forget_client] for forget_client in args.forget_clients]
    avg_forget_client_contribution = get_group_contribution(forget_client_contribution)
    retain_client_contribution = [avg_client_contributions[retain_client] 
                                  for retain_client in range(args.total_num_clients) 
                                  if retain_client not in args.forget_clients]
    avg_retain_client_contribution = get_group_contribution(retain_client_contribution)
    # unlearning, dampen the contributions/weights
    unlearned_global_weights = apply_dampening(global_model, avg_retain_client_contribution, avg_forget_client_contribution, 
                                               dampening_constant = args.dampening_constant, 
                                               dampening_upper_bound = args.dampening_upper_bound, 
                                               ratio_cutoff = args.ratio_cutoff)


    # evaluate the unlearned model
    test_acc_global, test_loss_global = test_local_model(unlearned_global_weights, test_dataloader, loss_fn, device)
    print(f"Test Accuracy for unlearned model : {test_acc_global*100}, Loss : {test_loss_global}")

    # check classwise accuracy
    print_classwise_accuracy(unlearned_global_weights, test_dataloader, device = args.device)

    # check attack accuracy
    if args.apply_backdoor:
        evaluate_backdoor_attack(unlearned_global_weights, backdoor_dataset,device = args.device)
    
    