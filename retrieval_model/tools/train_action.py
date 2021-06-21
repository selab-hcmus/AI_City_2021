import h5py, os, json, sys, shutil
import numpy as np 
import os.path as osp

sys.path.append(os.getcwd())

from coot import arguments_coot
from coot.configs_retrieval import ExperimentTypesConst, RetrievalConfig as Config
from coot.aic_dataset import create_retrieval_datasets_and_loaders
from coot.model_retrieval import RetrievalModelManager as ModelManager
from coot.trainer_retrieval import RetrievalTrainer as Trainer
from nntrainer import arguments, utils
from nntrainer.utils_torch import set_seed
from nntrainer.utils_yaml import load_yaml_config_file


EXP_TYPE = ExperimentTypesConst.RETRIEVAL

def setup_environment(cfg):
    # set seed
    if cfg.random_seed is not None:
        print(f"Set seed to {cfg.random_seed}")
        set_seed(cfg.random_seed, set_deterministic=False)  # set deterministic via config if needed

    # create dataset and dataloader
    if (cfg.dataset_train.preload_vid_feat or cfg.dataset_train.preload_text_feat or cfg.dataset_val.preload_vid_feat or
            cfg.dataset_val.preload_text_feat):
        cmd = "ulimit -n 100000"
        print(f"Run system command to avoid TooManyFiles error:\n{cmd}")
        os.system(cmd)

    pass

def setup_config():
    # ---------- Setup script arguments. ----------
    parser = utils.ArgParser(description=__doc__)
    arguments.add_default_args(parser)  # logging level etc.
    arguments.add_exp_identifier_args(parser)  # arguments to identify the experiment to run
    arguments.add_trainer_args(parser)  # general trainer arguments
    arguments.add_dataset_test_arg(parser)  # flag for dataset testing
    arguments_coot.add_dataloader_args(parser)  # feature preloading
    parser.add_argument("--load_model", type=str, default=None, help="Load model from file.")
    parser.add_argument("--uptrain", action="store_true", help="Uptrain or not")

    args = parser.parse_args()

    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config and dataset path given the script arguments
    path_data = arguments.update_path_from_args(args)
    config = arguments.update_config_from_args(config, args)
    config = arguments_coot.update_coot_config_from_args(config, args)

    # read experiment config dict
    cfg = Config(config, is_train=not args.validate and not args.test_dataset)
    if args.uptrain:
        cfg.dataset_train.split = 'val'

    setup_environment(cfg)
    return cfg, path_data, args, exp_group, exp_name


def main():
    # Setup everything
    cfg, path_data, args, exp_group, exp_name = setup_config()

    print(f'Path_data: {path_data}')
    
    path_action = osp.join(path_data, cfg.dataset_train.name, 'action_train_ohe.pkl')

    print('*'*20 + 'Init dataset')
    train_set, val_set, train_loader, val_loader = create_retrieval_datasets_and_loaders(cfg, path_action, path_data)
    run_name = f'{args.run_name}1'
    if args.uptrain:
        run_name = 'uptrain'
        print(f'Uptrain from checkpoint {args.load_model}')
        pass
    
    print()
    print('*'*20 + 'Create Model')    
    # create coot models
    model_mgr = ModelManager(cfg)
    load_best=False
    
    print()
    print('*'*20 + 'Create Trainer')
    # create trainer
    trainer = Trainer(
        cfg, model_mgr, exp_group, exp_name, run_name, len(train_loader), log_dir=args.log_dir,
        log_level=args.log_level, logger=None, print_graph=args.print_graph, reset=args.reset, load_best=load_best,
        load_epoch=args.load_epoch, load_model=args.load_model, inference_only=args.validate)

    print()
    print('*'*20 + 'Start Training')
    trainer.train_model(train_loader, val_loader)

if __name__ == '__main__':
    main()