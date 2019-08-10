import torch.utils.data
import importlib
from lib.utils.logging import setup_logging
logger = setup_logging(__name__)

class CustomerDataLoader():
    def __init__(self, opt):
        self.opt = opt
        self.dataset = create_dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchsize,
            shuffle=True if 'train' in opt.phase else False,
            num_workers=opt.thread)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchsize >= float("inf"):
                break
            yield data

def create_dataset(opt):
    dataset = find_dataset_lib(opt.dataset)()
    dataset.initialize(opt)
    logger.info("%s is created." % opt.dataset)
    return dataset


def find_dataset_lib(dataset_name):
    """
    Give the option --dataset [datasetname], import "data/datasetname_dataset.py"
    :param dataset_name: --dataset
    :return: "data/datasetname_dataset.py"
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
    if dataset is None:
        logger.info("In %s.py, there should be a class name that matches %s in lowercase." % (
        dataset_filename, target_dataset_name))
        exit(0)
    return dataset