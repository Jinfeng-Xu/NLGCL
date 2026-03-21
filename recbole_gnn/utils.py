import os
import pickle
import importlib
from logging import getLogger
from recbole.data.utils import load_split_dataloaders, create_samplers, save_split_dataloaders
from recbole.data.utils import create_dataset as create_recbole_dataset
from recbole.data.utils import data_preparation as recbole_data_preparation
from recbole.utils import set_color, Enum
from recbole.utils import get_model as get_recbole_model
from recbole.utils import get_trainer as get_recbole_trainer
from recbole.utils.argument_list import dataset_arguments

from recbole_gnn.data.dataloader import CustomizedTrainDataLoader, CustomizedNegSampleEvalDataLoader, CustomizedFullSortEvalDataLoader


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
    Returns:
        Dataset: Constructed dataset.
    """
    model_type = config['MODEL_TYPE']
    dataset_module = importlib.import_module('recbole_gnn.data.dataset')
    gen_graph_module_path = '.'.join(['recbole_gnn.model.general_recommender', config['model'].lower()])
    seq_module_path = '.'.join(['recbole_gnn.model.sequential_recommender', config['model'].lower()])
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    elif importlib.util.find_spec(gen_graph_module_path, __name__):
        dataset_class = getattr(dataset_module, 'GeneralGraphDataset')
    elif importlib.util.find_spec(seq_module_path, __name__):
        dataset_class = getattr(dataset_module, 'SessionGraphDataset')
    elif model_type == ModelType.SOCIAL:
        dataset_class = getattr(dataset_module, 'SocialDataset')
    else:
        return create_recbole_dataset(config)

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset
    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_submodule = [
        'general_recommender', 'sequential_recommender', 'social_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole_gnn.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        model_class = get_recbole_model(model_name)
    else:
        model_class = getattr(model_module, model_name)
    return model_class


def _get_customized_dataloader(config, phase):
    if phase == 'train':
        return CustomizedTrainDataLoader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == 'full':
            return CustomizedFullSortEvalDataLoader
        else:
            return CustomizedNegSampleEvalDataLoader


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.
    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.
    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    return recbole_data_preparation(config, dataset)


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name
    Args:
        model_type (ModelType): model type
        model_name (str): model name
    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('recbole_gnn.trainer'), model_name + 'Trainer')
    except AttributeError:
        return get_recbole_trainer(model_type, model_name)


class ModelType(Enum):
    """Type of models.

    - ``Social``: Social-based Recommendation
    """

    SOCIAL = 7
