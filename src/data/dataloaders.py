import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

DATA_ROOT = '/workspace/Dataset/Data'


def mnist(batch_size=100, pm=False):
    transf = [transforms.Resize(32),
              transforms.CenterCrop(32),
              transforms.ToTensor(),
              transforms.Normalize((0.5), (0.5))]
    if pm:
        transf.append(transforms.Lambda(lambda x: x.view(-1, 1024)))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_ROOT, train=False, transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    num_classes = 10

    return train_loader, val_loader, num_classes


def fashion_mnist(batch_size=100, pm=False):
    transf = [transforms.ToTensor()]
    transf.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DATA_ROOT + '/F_MNIST_data/', train=True, download=True, transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DATA_ROOT + '/F_MNIST_data/', train=False, transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    return train_loader, val_loader


def imagenet(augment=True, batch_size=100, classes=10):
    if classes == 10:
        data_dir = '/10_class_imagenet'
    else:
        data_dir = '/imagenet'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        logging += ' augmented'

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print(logging + ' IMAGENET.')
    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageNet(DATA_ROOT + data_dir, split='train',
                          transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageNet(DATA_ROOT + data_dir, split='val', transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = classes

    return train_loader, val_loader, num_classes


def cifar10(augment=True, normalization=True, batch_size=128, drop_last=False, data_root=None):
    if data_root == None:
        data_root = DATA_ROOT

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    logging = 'Using'

    trans = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    if augment:
        if normalization:
            trans.append(normalize)
        transform_train = transforms.Compose(trans)
        logging += ' augmented'
    else:
        trans = [
            transforms.ToTensor()
        ]
        if normalization:
            trans.append(normalize)
        transform_train = transforms.Compose(trans)

    trans = [
        transforms.ToTensor()
    ]
    if normalization:
        trans.append(normalize)
    transform_test = transforms.Compose(trans)

    print(logging + ' CIFAR 10.')
    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root + '/cifar10', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_root + '/cifar10', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar100(augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    print(logging + ' CIFAR 100.')
    kwargs = {'num_workers': 4, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DATA_ROOT + '/cifar100', train=True, download=True,
                          transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(DATA_ROOT + '/cifar100', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 100

    return train_loader, val_loader, num_classes
