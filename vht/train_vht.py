# %%
from vht.model.tracking_spree import FlameTracker as Tracker
from vht.data.video import VideoDataset
from vht.util.log import get_logger
from vht.util.video_to_dataset import Video2DatasetConverter

from argparse import ArgumentParser
from configargparse import ArgumentParser as ConfigArgumentParser
from pathlib import Path
from nha.spree_data.spree_dataset import SpreeDatasetModule
from tqdm import tqdm
logger = get_logger("vht", root=True)


def main():
    parser = ArgumentParser()
    parser = Tracker.add_argparse_args(parser)
    parser = SpreeDatasetModule.add_argparse_args(parser)
    parser = ConfigArgumentParser(parents=[parser], add_help=False)
    parser.add_argument("--config", default='/home/alaa/TriNHA/neural-head-avatars/deps/video-head-tracker/configs/tracking_spree.ini', is_config_file=True)
    parser.add_argument("--epochs", default=1)
    parser.add_argument("--batch_size", default=4)


    args = parser.parse_args()
    args_dict = vars(args)

    logger.info(f"Start tracking with the following configuration: \n {parser.format_values()}")

    # init datamodule
    data = SpreeDatasetModule(**args_dict)
    data.setup()    
    tracker = Tracker(**args_dict)

    tracker.predict_params
    for epoch in tqdm(range(args.epochs)):
        for batch in data.train_dataloader(args.batch_size):
            tracker.predict_params(batch)

if __name__ == "__main__":
    main()

# %%
