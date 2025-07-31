import logging

from evaluation.evaluator import get_metrics
from loggers.logging import Logger
from models.model_factory import ModelFactory
from dataset_loader.dataset_loader_factory import DatasetLoaderFactory
from feature_generator.feature_generator_factory import FeatureGeneratorFactory
from utils.experiment_io import get_run_id, get_run_dir, save_run_artifacts
from utils.config_handle import load_config
from utils.get_device import get_device
from utils.seed_all import seed_all

def main(config=None, data=None):
    if config is None:
        config = load_config(default_file_name="Gliner-ai4privacy")

    if 'run_id' not in config:
        run_id = get_run_id(config, [config['model']['name'], config['dataset']['name'], config['phase']])
        config['run_id'] = run_id
    
    run_id = config['run_id']
    run_dir = get_run_dir(run_id)

    save_run_artifacts(run_dir, config)

    # Setup logger
    logger = Logger(name="train_validation", log_file=f"{run_dir}/output.log", 
                    level=logging.DEBUG if 'debug' in config and config['debug'] else logging.INFO)

    # log run id
    logger.info("Initiating training and validation...")
    logger.info(f"[ RUN ID: {run_id} ]")

    seed = 0
    seed_all(seed)
    logger.debug(f"[ Using Seed : {seed} ]")

    # 1. Load the dataset
    if data is None:
        logger.debug("Loading data...")
        dataset_loader = DatasetLoaderFactory(config).get_dataset_loader(logger)
        feature_generator = FeatureGeneratorFactory(config).get_feature_generator(logger, dataset_loader)
        data = feature_generator.load_processed()
        logger.info("Data loaded successfully.")
    else:
        logger.info("Using provided data for training and validation.")

    # 3. Initializations
    logger.debug("Initializing components...")

    device = get_device()

    logger.info(f"Using device: {device}")

    # 3.1 Model
    logger.debug("Initializing model...")
    model_factory = ModelFactory(config, logger)
    model = model_factory.create_model()
    model = model.to(device)

    # 4. Execute
    logger.debug("Starting execution...")
    y_pred, y_true = model.evaluate(data)
    metrics = get_metrics(y_pred, y_true, logger=logger)
    logger.info(f"Metrics: {metrics}")
    logger.info("Execution completed.")

    # 6. Save run artifacts
    logger.debug("Saving run artifacts...")
    save_run_artifacts(run_dir, config, metrics=metrics)
    logger.info("Run artifacts saved.")
    
    logger.info("Validation completed.")

    del data, model, logger

    return metrics

if __name__ == "__main__":
    main()
