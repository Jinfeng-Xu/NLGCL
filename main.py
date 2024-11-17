import argparse
import logging
from logging import getLogger

from recbole.utils import init_logger, init_seed, set_color
from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

from trainer import MyTrainer


def run_single_model(args):
    # configurations initialization
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    if config['model'].lower() == "ncl":
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    else:
        trainer = MyTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = MyTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'model': config['model'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NLGCL', help='name of models')
    parser.add_argument('--dataset', type=str, default='yelp',
                        help='The datasets can be:')
    # 'yelp', 'pinterest', 'QB-video', 'alibaba'
    parser.add_argument('--config_files', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml'
    ]
    if args.config_files != '':
        args.config_file_list.extend(args.config_files.split(","))
    if args.dataset in ['yelp', 'pinterest', 'QB-video', 'alibaba']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')

    run_single_model(args)