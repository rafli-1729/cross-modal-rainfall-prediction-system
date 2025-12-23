import logging
from pathlib import Path
from src.dataset import merge_each_city, merge_all_cities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def main():
    verbose = True
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    RAW_PATH = Path(PROJECT_ROOT/'data/raw')
    PROCESS_PATH = Path(PROJECT_ROOT/'data/process')
    MERGE_PATH = PROCESS_PATH/'merge'

    logger.info("Merging yearly data (train)...")
    merge_each_city(
        input_root=RAW_PATH/'train',
        output_dir=MERGE_PATH/'train',
        verbose=verbose
    )

    logger.info("Merging yearly data (test)...")
    merge_each_city(
        input_root=RAW_PATH/'test',
        output_dir=MERGE_PATH/'test',
        verbose=verbose
    )

    logger.info("Merging all cities to single dataset (train)...")
    merge_all_cities(
        input_root=MERGE_PATH/'train',
        output_dir=PROCESS_PATH/'train.csv',
    )

    logger.info("Merging all cities to single dataset (test)...")
    merge_all_cities(
        input_root=MERGE_PATH/'test',
        output_dir=PROCESS_PATH/'test.csv',
    )

if __name__ == '__main__':
    main()