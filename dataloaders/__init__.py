from torch.utils.data import DataLoader
from dataloaders import face_dataset
from torch.utils.data.sampler import *
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses

    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    # count is counts of each cls id eg. cls1 has 100 images  cls2 has 200 images
    
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])

    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    # weight is weight of each images eg. cls1 images weights are same

    return weight
    
def make_data_loader(batch_size):

    INPUT_SIZE = [112, 112]
    DATA_ROOT = '/home/data/CASIA/CASIA-WebFace-Mask-Aligned'

    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
    IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_RGB_MEAN,
                             std=IMAGE_RGB_STD),
    ])

    dataset_train = datasets.ImageFolder(DATA_ROOT, train_transform)

    # create a weighted random sampler to process imbalanced data
    # dataset.imgs[(‘data/dogcat_2/cat/cat.12484.jpg’, 0), (‘data/dogcat_2/cat/cat.12485.jpg’, 0), (‘data/dogcat_2/dog/dog.12499.jpg’, 1)]
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, sampler=sampler, pin_memory=True,
        num_workers=4, drop_last=True)

    NUM_CLASS = len(train_loader.dataset.classes)

    return train_loader, NUM_CLASS
