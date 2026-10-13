"""
ExamDataModuleBase.py

MIL = Multiple Instance Learning, refers to a weakly supervised learning approach
where the labels are only available at the bag level rather than at the instance level,
each bag consisting of multiple instances, not all of which need have the same label and
whose labels are generally not known.

This module defines the ExamDataModuleBase, a foundational class for handling
FAMLI datasets in PyTorch Lightning. It encapsulates common logic for data loading,
initialization, and setup, which is then extended by specialized DataModules for
training and inference.

This Dataset module is used to dish out samples for training and validation
at an exam level (not video level). This is appropriate for models that need
exam level information during training. For example, models that use multiple
instance learning.

This base class is designed to be subclassed and is not intended for direct use.

Four parameters are used to control the way frames are sampled:
- frame_sampling      - Temporal sampling of frames, within a video
                        or the entire exam of frames.
- use_known_combos    - Filter videos by a fixed set of known_combos,
                        suitable for inference
- random_known_combos - Randomly sample from the known_combos ON-THE-FLY,
                        suitable for training as a form of data augmentation.
- allow_biometric     - Use biometric sweeps when there aren't enough blind sweeps
                        This parameter is shared between DataModule and Dataset.
- strict_known_combos - Whether to use strict definition of known_combos
                        This parameter is shared between DataModule and Dataset.

**NOTE**  use_known_combos and random_known_combos should not both be set to True;
Either one can be set to True, or both set to False, but not both set to True.

Author: Courosh Mehanian
        Daniel Shea
        Olivia Zahn
        Sourabh Kulhare

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
# Python standard library
import os
import pandas as pd
from typing import Callable, Union, List, Tuple

# External imports
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

# Internal imports
from ghlobus.utilities.paths import v9_5fold_dir
from ghlobus.utilities.constants import DIST_DIR
from ghlobus.utilities.sweep_utils import filter_by_known_combos

# Local imports
from ghlobus.data.ExamDatasetFrame import ExamDatasetFrame
from ghlobus.utilities.sample_utils import compute_equal_sampling_weights_by_trimester


class ExamDataModuleBase(LightningDataModule):
    """
    A base class for Exam-based DataModules, providing common functionality.
    """
    def __init__(self,
                 dataset_dir: os.PathLike = v9_5fold_dir,
                 distribution: Union[None, str, int] = None,
                 data_file_template: str = '{}.csv',
                 transforms: List[Callable] = (),
                 augmentations: List[Callable] = (),
                 batch_size: int = 2,
                 num_workers: int = 32,
                 bag_size: int = 300,
                 mil_format: str = 'frame',
                 channels: int = 3,
                 frames: int = 50,
                 frame_sampling: str = 'matern',
                 use_known_combos: bool = False,
                 random_known_combos: bool = False,
                 allow_biometric: bool = False,
                 strict_known_combos: bool = False,
                 frames_or_channel_first: str = 'frames',
                 use_stratified_sampler: bool = False,
                 use_inference_val_dataset: bool = False,
                 upsample: bool = True,
                 max_replicates: int = 10,
                 balance_ga: bool = False,
                 image_dims: Union[int, Tuple[int, int]] = (256, 256),
                 ) -> None:
        """
        Exam-based learning DataModule for TWIN Detection, e.g., using
            Multiple Instance Learning (MIL) or other methods.

        This DataModule is a front-end for a Dataset object that loads frames from videos
        and organizes them into bags for Multiple Instance Learning (MIL). It supports both
        frame-level and video-level instances, with various sampling strategies for frame selection.

        Parameters:
            dataset_dir: os.PathLike      - Root directory containing data split definition files
            distribution: int             - Integer indicating the distribution ID to use
            data_file_template: str       - Template string describing the data file name
            transforms: List[Callable]    - List of functions to transform the data prior
                                            to getting passed to the model
            augmentations: List[Callable] - List of functions to transform the data prior
                                            to getting passed to the model, train time only
            batch_size: int               - Size of batch for dataloaders
            num_workers: int              - Number of workers for dataloaders
            bag_size: int                 - Size of bags for dataloaders
            mil_format: str               - MIL approach: bag=exam and instance= 'video' or 'clip' or 'frame'
            channels: int                 - Number of channels for output (keep 1 or repeat to 3)
            frames: int                   - Number of sampled frames in each video (mil_format='video')
                                            Number of half-sampled frames in each clip (mil_format='clip')
            frame_sampling: str           - Either 'random', 'jitter', 'uniform', or 'matern'
            frames_or_channel_first: str  - Either 'frames' or 'channel'
            use_known_combos: bool        - Whether to filter sweeps by KNOWN_COMBOS (**FIXED**)
            random_known_combos: bool     - Whether to randomly sample from KNOWN_COMBOS (**ON-THE-FLY**)
            allow_biometric: bool         - Whether to use biometric sweeps when insufficient KNOWN_COMBOS
            strict_known_combos: bool     - Whether to use strict definition of KNOWN_COMBOS
            upsample:                     - Whether to upsample exams from the minority class
            max_replicates: int           - Maximum number of times to replicate an exam
            balance_ga: bool              - Whether to balance GA when upsampling exams
            image_dims: int or (int,int)  - Size of output image for inference. CenterCrop
                                            is performed on the data to the given image dims.
        """
        # Call the parent constructor
        super().__init__()

        # See if distribution subfolder is specified
        if distribution is None:
            self.data_dir = dataset_dir
        else:
            self.data_dir = os.path.join(dataset_dir, DIST_DIR, str(distribution))

        # Store data_file_template
        self.data_file_template = data_file_template

        # Format of the MIL learning 'frame', 'clip', or 'video'
        if mil_format not in ['frame', 'clip', 'video']:
            raise ValueError(f"MIL format {mil_format} is not recognized")
        self.mil_format = mil_format

        # Sampling parameters
        if use_known_combos and random_known_combos:
            # These are mutually exclusive:
            #   - use_known_combos filters at instantiation and remains fixed
            #   - random_known_combos selects randomly from known_combos at runtime
            raise ValueError("Both use_known_combos and random_known_combos cannot be True.")
        self.use_known_combos = use_known_combos
        self.allow_biometric = allow_biometric
        self.strict_known_combos = strict_known_combos

        # Set up Dataset keyword arguments
        self.ds_kwargs = {
            'bag_size': bag_size,
            'mil_format': mil_format,
            'channels': channels,
            'frames': frames,
            'frame_sampling': frame_sampling,
            'frames_or_channel_first': frames_or_channel_first,
            'random_known_combos': random_known_combos,
            'allow_biometric': allow_biometric,
            'strict_known_combos': strict_known_combos,
            'upsample': upsample,
            'max_replicates': max_replicates,
            'balance_ga': balance_ga,
            'image_dims': image_dims,
        }
        self.transforms = list(transforms)
        self.augmentations = list(augmentations)

        # then Set up DataLoader options
        self.dl_options = {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }

        # Attributes for experiment configuration
        self.use_stratified_sampler = use_stratified_sampler
        self.use_inference_val_dataset = use_inference_val_dataset

        # Store dataset names, via stages
        self.allowed_stages = dict()

        # Set up dictionary container for datasets
        self.dfs = dict()

    def setup(self, stage: str) -> None:
        """
        Setup the inputs for the xxx_dataloader() functions.
            This ignores the 'stage' parameter for now,
            and sets up all dataloader/dataset params.

        Parameters:
            stage: str     - string indicating the stage parameter typically passed to or
                             implicitly determined by the Trainer.
        """
        # Check for valid stage
        if stage not in self.allowed_stages:
            message = f"{self.__class__.__name__} can only be set up for stage: "
            message += f"{list(self.allowed_stages.keys())}, but got: {stage}."
            raise ValueError(message)

        # Gather the datasets to load based on stage, defined in the children classes
        # Dataset types (generally) are: ['train', 'val', 'test']
        datasets = self.allowed_stages.get(stage, None)
        if not datasets:
            message = f"No datasets defined for stage '{stage}' in {self.__class__.__name__}."
            raise ValueError(message)

        # Load and store each dataset .csv spreadsheet
        for dataset in datasets:
            # Create data file name
            datafile_name = self.data_file_template.format(dataset)
            # Read the data file
            df = pd.read_csv(os.path.join(self.data_dir, datafile_name))
            # Keep only the unique filenames, and sort by exam_dir
            df.drop_duplicates(subset='filename', inplace=True, ignore_index=True)
            df.sort_values(by=['exam_dir', 'filename'], inplace=True, ignore_index=True)
            # Whether to filter by known_combos
            if self.use_known_combos:
                df = filter_by_known_combos(
                    df,
                    allow_biometric=self.allow_biometric,
                    strict_known_combos=self.strict_known_combos,
                )
            # Check for empty dataframe
            if df.shape[0] == 0:
                raise ValueError(f"Dataframe for {dataset} is empty after filtering by known_combos.")
            # Store the dataframe to `dfs` list
            df.sort_values(by=['exam_dir', 'filename'], inplace=True, ignore_index=True)
            df.reset_index(drop=True, inplace=True)
            self.dfs[dataset] = df

    def _create_train_dataloader(self, dataset_name: str) -> DataLoader:
        """
        Set up the train DataLoader for the given dataset name.

        Args:
            dataset_name (str): Name of the dataset to set up ('train', 'val', or 'test').

        Returns:
            DataLoader: PyTorch DataLoader instance for the specified dataset.
        """
        # Step 1. Gather the Dataset keyword arguments
        # Initialize kwargs dictionary, always using transforms
        df = self.dfs[dataset_name]
        ds_kwargs = {
            'df': df,
            'transforms': self.transforms,
        }
        ds_kwargs.update(self.ds_kwargs)
        if dataset_name == 'train':
            ds_kwargs['augmentations'] = self.augmentations
            ds_kwargs['mode'] = 'train'
        elif dataset_name == 'val':
            ds_kwargs['augmentations'] = []
            ds_kwargs['mode'] = 'val'

        # Step 2. Create the Dataset
        dataset = self._create_dataset(**ds_kwargs)

        # Step 3. Add a sampler, if using it. (Optional)
        if self.use_stratified_sampler and dataset_name == 'train':
            weights = compute_equal_sampling_weights_by_trimester(df)
            sampler = torch.utils.data.WeightedRandomSampler(weights=torch.DoubleTensor(weights),
                                                             num_samples=len(weights))
        else:
            # Default
            sampler = None

        # Step 4. Create the DataLoader
        dataloader = DataLoader(dataset, sampler=sampler, **self.dl_options)

        return dataloader

    def _create_inference_dataloader(self, dataset_name: str) -> DataLoader:
        """
        Set up a DataLoader for the given dataset name.

        Args:
            dataset_name (str): Name of the dataset to set up ('train', 'val', or 'test').

        Returns:
            DataLoader: PyTorch DataLoader instance for the specified dataset.
        """
        # Step 1. Gather the Dataset keyword arguments
        # Initialize kwargs dictionary, note only using transforms, no augmentations for inference
        ds_kwargs = {
            'df': self.dfs[dataset_name],
            'transforms': self.transforms,
            'augmentations': [],
        }
        # Update with args specified in constructor
        ds_kwargs.update(self.ds_kwargs)
        ds_kwargs['mode'] = 'test'

        # Step 2. Create the Dataset
        dataset = self._create_dataset(**ds_kwargs)

        # Step 3. Copy the keyword parameters for DataLoader,
        dl_options = self.dl_options.copy()
        # but replace the batch_size with 1 (since each video has different length)
        dl_options['batch_size'] = 1

        # Step 4. Create the DataLoader
        dataloader = DataLoader(dataset, sampler=None, **dl_options)

        return dataloader

    def _create_dataset(self, **kwargs):
        """Import optional MIL datasets only when their format is selected."""
        if self.mil_format == 'frame':
            return ExamDatasetFrame(**kwargs)
        if self.mil_format == 'clip':
            from ghlobus.data.ExamDatasetClip import ExamDatasetClip
            return ExamDatasetClip(**kwargs)
        if self.mil_format == 'video':
            from ghlobus.data.ExamDatasetVideo import ExamDatasetVideo
            return ExamDatasetVideo(**kwargs)
        raise ValueError(f"MIL format {self.mil_format} is not recognized.")
