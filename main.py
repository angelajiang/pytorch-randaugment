from torchvision.transforms import transforms
#from RandAugment import RandAugment
import RandAugment

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Add RandAugment with N, M(hyperparameter)
N = 3
M = 1
transform_train.transforms.insert(0, RandAugment.RandAugment(N, M))

RandAugment.augmentations.ShearX
