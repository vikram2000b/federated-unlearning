from torchvision import datasets, transforms

def get_dataset(dataset_name):
    """
    Get the dataset.
    Args:
        dataset_name: string, name of the dataset
    Returns:
        train_dataset: torch.utils.data.Dataset, training dataset
        test_dataset: torch.utils.data.Dataset, testing dataset
        num_classes: int, number of classes in the dataset
    """
    if dataset_name == 'cifar10':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                        transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
        return train_dataset, test_dataset, 10
    elif dataset_name == 'mnist':
        data_dir = './data/mnist/'
        apply_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                    transform=apply_transform)
        return train_dataset, test_dataset, 10
    

def create_dirichlet_data_distribution(dataset, num_clients, num_classes):
    """
    Create a dirichlet data distribution.
    Args:
        dataset: torch.utils.data.Dataset, dataset
        num_clients: int, number of clients
        num_classes: int, number of classes
    Returns:
        client_groups: dict, client groups
    """
    raise NotImplementedError()

def create_iid_data_distribution(dataset, num_clients, num_classes):
    """
    Create an iid data distribution.
    Args:
        dataset: torch.utils.data.Dataset, dataset
        num_clients: int, number of clients
        num_classes: int, number of classes
    Returns:
        client_groups: dict, client groups
    """
    raise NotImplementedError()
