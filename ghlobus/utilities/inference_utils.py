"""
inference_utils.py

A collection of useful functions for inference-related tasks.

Author: Dan Shea
        Courosh Mehanian
        Olivia Zahn
        Wenlong Shi

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import glob
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pydicom import dcmread
from functools import partial
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Optional, Any

from torch.nn import Linear
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from ghlobus.models.TvCnn import TvCnn
from ghlobus.utilities.biometry_utils import efw_hadlock_4component
from ghlobus.utilities.plot_utils import plot_exam_attention_scores
from ghlobus.utilities.plot_utils import plot_attention_scores
from ghlobus.utilities.sample_utils import inference_subsample
from ghlobus.utilities.data_utils import prepare_frames
from ghlobus.utilities.data_utils import preprocess_video
from ghlobus.utilities.constants import VERY_LARGE_NUMBER
from ghlobus.models.TvCnnFeatureMap import TvCnnFeatureMap
from ghlobus.models.BasicAdditiveAttention import MultipleAdditiveAttention
from ghlobus.models.Cnn2RnnClassifier import Cnn2RnnClassifier
from ghlobus.models.Cnn2RnnRegressor import Cnn2RnnRegressor
from ghlobus.models.TvMilCnn import TvMilCnn
from ghlobus.models.MilAttention import MilAttention
from ghlobus.models.MultiClassifier import MultiClassifier


from ghlobus.utilities.normalization import (
    LGA_MEAN, LGA_STD, AC_MEAN, AC_STD, HC_MEAN, HC_STD,
    FL_MEAN, FL_STD, BPD_MEAN, BPD_STD,
)


# Inference functions
def load_Cnn2RnnRegressor_model(modelpath: os.PathLike,
                                device: str,
                                cnn_name: str = 'MobileNet_V2',
                                weights_name: str = 'DEFAULT',
                                mode: str = 'GA'
                                ) -> LightningModule:
    """
    Load a Cnn2RnnRegressor model from a checkpoint for inference.
    ! Note: Sets the `report_intermediates` attribute to True, which returns
    ! intermediate results from the network

    Parameters:
        modelpath (os.PathLike):       Path to the model checkpoint.
        device (str):                  Device to load the model on.
        cnn_name (str, optional):      Name of the CNN. Defaults to 'MobileNet_V2'.
        weights_name (str, optional):  Name of the weights. Defaults to 'DEFAULT'.
        mode (str, optional):          Whether to run in GA or EFW mode. Defaults to 'GA'.

    Returns:
        LightningModule: The loaded model.
    """
    if mode == 'GA':
        model = Cnn2RnnRegressor.load_from_checkpoint(
            checkpoint_path=f'{modelpath}',
            map_location=f'{device}',
            cnn=TvCnn(tv_model_name=cnn_name, tv_weights_name=weights_name),
        )
    elif mode == 'EFW':
        model = Cnn2RnnRegressor.load_from_checkpoint(
            checkpoint_path=f'{modelpath}',
            map_location=f'{device}',
            cnn=TvCnn(tv_model_name=cnn_name, tv_weights_name=weights_name),
            rnn=MultipleAdditiveAttention(
                input_dim=1000, attention_dim=16, num_modules=8),
            regressor=Linear(in_features=8000, out_features=4),
        )
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    # Set the .return_intermediates attribute to 'True', so that .forward() returns
    # all intermediate states of importance (feature vectors, context vectors, attention scores)
    model.report_intermediates = True
    # Set model to eval mode, and freeze weights
    model.eval()
    model.freeze()

    return model


def load_Cnn2RnnClassifier_model(modelpath: os.PathLike,
                                 device: str,
                                 cnn_name: str = "MobileNet_V2",
                                 weights_name: str = 'IMAGENET1K_V2',
                                 cnn_layer_id: int = 18,
                                 ) -> LightningModule:
    """
    Load a Cnn2RnnClassifier model from a checkpoint for inference.
    ! Note: Sets the `report_intermediates` attribute to True, which returns
    ! intermediate results from the network

    Parameters:
        modelpath (os.PathLike):       Path to the model checkpoint.
        device (str):                  Device to load the model on.
        cnn_name (str, optional):      Name of the CNN. Defaults to 'MobileNet_V2'.
        weights_name (str, optional):  Name of the weights. Defaults to 'IMAGENET1K_V2'.
        cnn_layer_id (int, optional):  Index of the layer to extract features from. Defaults to 18.

    Returns:
        LightningModule: The loaded model.
    """
    model = Cnn2RnnClassifier.load_from_checkpoint(
        checkpoint_path=f'{modelpath}',
        map_location=f'{device}',
        cnn=TvCnnFeatureMap(cnn_name=cnn_name,
                            cnn_weights_name=weights_name,
                            cnn_layer_id=cnn_layer_id),
    )

    # Set the .return_intermediates attribute to 'True', so that .forward() returns
    # all intermediate states of importance (feature vectors, context vectors, attention scores)
    model.report_intermediates = True
    # Set model to eval mode, and freeze weights
    model.eval()
    model.freeze()

    return model


def load_TWIN_model(modelpath: os.PathLike,
                    device: str,
                    cnn_name: str = 'MobileNet_V2',
                    weights_name: str = 'IMAGENET1K_V2',
                    ) -> LightningModule:
    """
    Load a Cnn2RnnClassifier model for TWIN (multiple gestation) classification
    from a checkpoint for inference. Uses the iMIL (Multiple Instance Learning)
    architecture with TvMilCnn, MilAttention, and MultiClassifier components.

    ! Note: Sets the `report_intermediates` attribute to True, which returns
    ! intermediate results from the network

    Parameters:
        modelpath (os.PathLike):       Path to the model checkpoint.
        device (str):                  Device to load the model on.
        cnn_name (str, optional):      Name of the CNN. Defaults to 'MobileNet_V2'.
        weights_name (str, optional):  Name of the weights. Defaults to 'IMAGENET1K_V2'.

    Returns:
        LightningModule: The loaded model.
    """
    # Create the iMIL architecture components matching the training configuration
    cnn = TvMilCnn(
        tv_model_name=cnn_name,
        tv_weights_name=weights_name,
        mil_format='frame',
        pretrained_path=None,
    )
    rnn = MilAttention(
        input_dim=1000,
        embedding_dim=[512, 256],
        attention_dim=64,
    )
    classifier = MultiClassifier(
        in_features=256,
        num_classes=2,
    )

    model = Cnn2RnnClassifier.load_from_checkpoint(
        checkpoint_path=f'{modelpath}',
        map_location=f'{device}',
        cnn=cnn,
        rnn=rnn,
        classifier=classifier,
    )

    # Set the .return_intermediates attribute to 'True', so that .forward() returns
    # all intermediate states of importance (feature vectors, context vectors, attention scores)
    model.report_intermediates = True
    # Set model to eval mode, and freeze weights
    model.eval()
    model.freeze()

    return model


def is_loadable_dicom(filename: os.PathLike) -> bool:
    """
    Function that tests if a given filename can be read using PyDicom `dcmread` function.

    Parameters:
        filename (os.PathLike):  The name of the file to load.

    Returns:
        bool:                    True if file was successfully loaded, False otherwise.
    """
    try:
        with dcmread(filename, stop_before_pixels=True, force=True) as dcm:
            if int(dcm.NumberOfFrames) > 1:
                return True
    except Exception:
        return False


def get_loadable_dicom_files(directory: os.PathLike) -> List[str]:
    """
    Returns a list of files in the given directory that can be successfully loaded.

    Parameters:
        directory (os.PathLike):  The directory to check for loadable files.

    Returns:
        List[str]:                A list of filenames that can be successfully loaded.
    """
    # get list of items in directory
    files_in_directory = glob.glob(os.path.join(directory, "*"))
    # include them if they are loadable dicom files
    loadable_files = [x for x in files_in_directory if
                      os.path.isfile(x) and is_loadable_dicom(x)]

    return loadable_files


def get_dicom_frames(dicompath: os.PathLike,
                     device: str = 'cpu',
                     mode: str = 'GA',
                     pdx: float = None) -> torch.Tensor:
    """
    Process the input DICOM and convert the frames into model-ready torch.Tensor.

    ! NOTE: Uses parameters that are default in our `ingestion` pipeline, typically used to
    ! prepare frames and save them as `.pt` PyTorch files. In this case, the frames
    ! are returned directly and subsequently processed further instead of saving the
    ! data to a .pt file.

    ? NOTE - This function scales data per the model training pipeline;
    ? Any changes to the scaling will give invalid results for model output.

    Parameters:
        dicompath (os.PathLike):  Path to the DICOM file.
        device (str):             Device to send the frames to for inference.
        mode (str):               Whether to run in GA or FP mode.
        pdx (float):              The physical delta X value from the DICOM metadata, or supplied to the function.

    Returns:
        torch.Tensor:             The processed frames.
    """
    is_dicom = is_loadable_dicom(dicompath)
    # stack file-specific information for passing to `preprocess_video` function
    file_info = pd.Series({
        'in_filepath': dicompath,
        'file_type': 'dcm' if is_dicom else 'mp4',
        'project': dicompath.split(os.path.sep)[-3],
        'exam_dir': dicompath.split(os.path.sep)[-2],
        'tag': 'Unknown',
        # Note: PDX gets extracted automatically if it is a DICOM, but
        # must be supplied if it is an MP4
        'pdx': None if is_dicom else pdx,
    })

    # Process DICOM with in-memory ingestion `preprocess_video` function
    frames = preprocess_video(file_info,
                              out_dir=None,
                              img_size=288,
                              channels=1,
                              dtype='uint8',
                              alpha=0.075,
                              min_frames=0,
                              doppler_rgb_thresh=VERY_LARGE_NUMBER,
                              doppler_ybr_thresh=VERY_LARGE_NUMBER,
                              doppler_pixel_thresh=VERY_LARGE_NUMBER,
                              raw=None,
                              jpg=None,
                              pt=None,
                              force_dcm_load=True,
                              )

    # Convert frames into model-ready torch.Tensor.
    if mode.upper() == 'GA' or mode.upper() == 'EFW':
        frames = prepare_frames(
            frames,
            channels=3,
        )
    elif mode.upper() == 'FP':
        frames = prepare_frames(
            frames,
            channels=3,
            subsample_method=inference_subsample,
        )
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    # Add a batch dimension to the frames Tensor.
    frames = frames.unsqueeze(0)
    # Send the data to the target device
    frames = frames.to(device)

    return frames


def run_inference(model: LightningModule,
                  X: torch.Tensor,
                  mode: str = 'GA',
                  lmean: float = LGA_MEAN,
                  lstd: float = LGA_STD,
                  ) -> Tuple[np.ndarray]:
    """
    Pass X through the model, and catch the return values.

    ? Because `model.return_intermediates` is `True`, the model
    ? returns the prediction (y_hat) in addition to the frame
    ? feature vectors (outputs from the CNN), context vectors (from the RNN)
    ? and attention scores (post-softmax) for the frames.

    Parameters:
        model (LightningModule):  The model to run inference on.
        X (torch.Tensor):         The input data.
        mode (str):               Whether to run in GA or FP mode or EFW mode.
        lmean (float):            The mean of the log values for rescaling.
        lstd (float):             The std of the log values for rescaling.

    Returns:
        Tuple[np.ndarray]:        The result of the inference.
    """
    # Pass the data forward through the model
    result = model(X)
    # Detach and convert to numpy all result Tensor values
    if mode.upper() == 'GA':
        # Create the rescale function for the log GA values
        rescale_log_ga = get_rescale_log_value_func(lmean, lstd)
        # Unpack the result
        result = detach_and_convert_tensors(model, result)
        y_hat, frame_features, context_vectors, attention_scores = result
        # Rescale the y_hat predicted log GA to GA in Days
        y_hat_days = rescale_log_ga(y_hat)

        return y_hat_days, y_hat, frame_features, context_vectors, attention_scores
    elif mode.upper() == 'FP':
        # Unpack the result
        result = detach_and_convert_tensors(model, result)
        y_hat, frame_features, context_vectors, auxiliary_vectors, logits = result
        # get maximum softmax output
        pred = np.argmax(y_hat, axis=-1)

        return pred, y_hat, frame_features, context_vectors, auxiliary_vectors, logits
    elif mode.upper() == 'EFW':
        # Unpack the result
        result = detach_and_convert_tensors(model, result)
        y_hat, frame_features, context_vectors, attention_scores = result
        # Rescale the y_hat values to EFW biometric values
        log_ac, log_fl, log_hc, log_bpd = y_hat
        # Each biometric is rescaled individually using the BioNormLogRescale class
        ac_pred = rescale_log_value(log_ac, AC_MEAN, AC_STD)
        fl_pred = rescale_log_value(log_fl, FL_MEAN, FL_STD)
        hc_pred = rescale_log_value(log_hc, HC_MEAN, HC_STD)
        bpd_pred = rescale_log_value(log_bpd, BPD_MEAN, BPD_STD)
        efw_pred = efw_hadlock_4component(bpd_pred, ac_pred, hc_pred, fl_pred)
        scaled_y_hat = (ac_pred, fl_pred, hc_pred, bpd_pred, efw_pred)
        y_hat = (log_ac, log_fl, log_hc, log_bpd)
        return scaled_y_hat, y_hat, frame_features, context_vectors, attention_scores
    else:
        raise ValueError(f"Mode {mode} not recognized.")


def detach_and_convert_tensors(model: LightningModule,
                               tensors: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Sends the torch.Tensor's to 'cpu' device, detaches from gradient tape, and
    converts to numPy. Useful for converting outputs from the model to numPy values.

    Parameters:
        model (LightningModule):       The model the tensors are associated with.
        tensors (List[torch.Tensor]):  The list of tensors to convert.

    Returns:
        List[np.ndarray]: The converted tensors.
    """
    # If the model is not run on CPU, transfer the tensors to CPU first
    if model.device != 'cpu':
        tensors = [x.cpu() for x in tensors]
    # Detach variables from gradient tape and convert to numpy
    tensors = [x.detach().numpy() for x in tensors]

    return tensors


def rescale_log_value(log_value, mean, std):
    """
    Returns a function that rescales a log value to the original scale.

    Parameters:
        log_value (np.ndarray):  Logarithmic value(s).
        mean (float):            Mean of the log values.
        std (float):             Standard deviation of the log values.

    Returns:
        np.ndarray:             Rescaled value(s).
    """
    # un-standardize by multiplying by std and adding mean
    # then exponentiate (e^x) to get GA in days
    return np.exp((log_value * std) + mean)


def get_rescale_log_value_func(mean, std):
    """
    Returns a function that rescales a log value to the original scale.

    Parameters:
        mean (float):  Mean of the log values.
        std (float):   Standard deviation of the log values.

    Returns:
        Callable:      Rescale function.
    """
    return partial(rescale_log_value, mean=mean, std=std)


class PredictionProcessor(ABC):
    """
    Abstract base class for prediction processors.
    """

    @abstractmethod
    def apply(self, value: Union[float, List[float]]) -> float:
        """
        Apply the processing to the given value.

        Parameters
        ----------
        value : float
            The value to be processed.

        Returns
        ----------
        float
            The processed value.
        """
        raise NotImplementedError(
            "Subclasses must implement the `apply` method.")


class NormLogRescale(PredictionProcessor):
    """
    A class used to undo normalization and log scaling.

    Attributes
    ----------
    mean : float
        The mean value used for rescaling.
    std : float
        The standard deviation value used for rescaling.
    """

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def apply(self, value: float) -> float:
        """
        Undo normalization and log scaling.

        Parameters
        ----------
        value : float
            The value to be processed.

        Returns
        ----------
        float
            The processed value.
        """
        return rescale_log_value(value, self.mean, self.std)


class BioNormLogRescale(PredictionProcessor):
    """
    A class used to undo normalization and log scaling for biometric log values and calculate the fetal weight.

    Attributes
    ----------
    ac_mean : float
        The mean value used for rescaling the abdominal circumference.
    ac_std : float
        The standard deviation value used for rescaling the abdominal circumference.
    hc_mean : float
        The mean value used for rescaling the head circumference.
    hc_std : float
        The standard deviation value used for rescaling the head circumference.
    fl_mean : float
        The mean value used for rescaling the femur length.
    fl_std : float
        The standard deviation value used for rescaling the femur length.
    bpd_mean : float
        The mean value used for rescaling the biparietal diameter.
    bpd_std : float
        The standard deviation value used for rescaling the biparietal diameter.
    """

    def __init__(self,
                 ac_mean: float = AC_MEAN,
                 ac_std: float = AC_STD,
                 hc_mean: float = HC_MEAN,
                 hc_std: float = HC_STD,
                 fl_mean: float = FL_MEAN,
                 fl_std: float = FL_STD,
                 bpd_mean: float = BPD_MEAN,
                 bpd_std: float = BPD_STD) -> None:
        self.ac_mean = ac_mean
        self.ac_std = ac_std
        self.hc_mean = hc_mean
        self.hc_std = hc_std
        self.fl_mean = fl_mean
        self.fl_std = fl_std
        self.bpd_mean = bpd_mean
        self.bpd_std = bpd_std

    def apply(self, value: List[float]) -> float:
        """
        Undo normalization and log scaling for the given biometric values and calculate the fetal weight using Hadlock formula.

        Parameters
        ----------
        value : List[float]
            The biometric values to be processed.

        Returns
        ----------
        float
            The processed fetal weight.
        """
        assert len(value) == 4, "The input list must have 4 elements."
        ac, fl, hc, bpd = value
        ac = rescale_log_value(ac, self.ac_mean, self.ac_std)
        fl = rescale_log_value(fl, self.fl_mean, self.fl_std)
        hc = rescale_log_value(hc, self.hc_mean, self.hc_std)
        bpd = rescale_log_value(bpd, self.bpd_mean, self.bpd_std)
        efw_pred = efw_hadlock_4component(bpd, ac, hc, fl)
        return efw_pred


bio_rescale = BioNormLogRescale(
    ac_mean=AC_MEAN,
    ac_std=AC_STD,
    hc_mean=HC_MEAN,
    hc_std=HC_STD,
    fl_mean=FL_MEAN,
    fl_std=FL_STD,
    bpd_mean=BPD_MEAN,
    bpd_std=BPD_STD,
)


def enumerate_dicom_files(dicom: Union[None, List[str]] = None,
                          examdir: Union[None, str] = None) -> List[str]:
    """
    Determine the list of DICOM files to analyze in this script.
    Two usage cases:
        dicom is list of dicompaths, examdir is None
        dicom is None , examdir points to folder of dicom files.

    Parameters:
        dicom (List[str]):      List of DICOM files.
        examdir (os.PathLike):  Directory containing DICOM files.

    Returns:
        List[str]:              List of DICOM files.
    """
    # Initialize an empty list of DICOM file paths
    dicomlist = []
    # if `dicom` is provided, use the list of DICOM files
    # if the DICOM file is loadable.
    if dicom:
        dicomlist = [
            dcmpath for dcmpath in dicom if is_loadable_dicom(dcmpath)]
    elif examdir:
        dicomlist = get_loadable_dicom_files(examdir)
    # Check to ensure at least one DICOM found
    if len(dicomlist) == 0:
        raise ValueError("No DICOM files found!")
    # Return the final list
    return dicomlist


def enumerate_mp4_files(mp4: Union[None, List[str]] = None,
                        examdir: Union[None, str] = None) -> List[str]:
    """
    Determine the list of MP4 files to analyze in this script.
    Two usage cases:
        mp4 is list of mp4 paths, examdir is None
        mp4 is None, examdir points to folder of mp4 files.

    Parameters:
        mp4 (List[str]):      List of MP4 files.
        examdir (os.PathLike):  Directory containing MP4 files.

    Returns:
        List[str]:              List of MP4 files.
    """
    mp4list = []
    if mp4:
        mp4list = [mp4path for mp4path in mp4 if os.path.isfile(
            mp4path) and mp4path.lower().endswith('.mp4')]
    elif examdir:
        files_in_directory = glob.glob(os.path.join(examdir, "*.mp4"))
        mp4list = [x for x in files_in_directory if os.path.isfile(x)]
    if len(mp4list) == 0:
        raise ValueError("No MP4 files found!")
    return mp4list


def enumerate_media_files(files: Union[None, List[str]] = None,
                          examdir: Union[None, str] = None) -> List[str]:
    """
    Combine the lists from enumerate_dicom_files and enumerate_mp4_files.
    Accepts optional lists of dicom and mp4 files, or a directory to search
    for both types of files.

    Parameters:
        files (List[str], optional): List of files.
        examdir (os.PathLike, optional): Directory containing DICOM
        and/or MP4 files.

    Returns:
        List[str]: Combined list of DICOM and MP4 files.
    """
    try:
        dicom_files = enumerate_dicom_files(dicom=files, examdir=examdir)
    except ValueError:
        dicom_files = []
    try:
        mp4_files = enumerate_mp4_files(mp4=files, examdir=examdir)
    except ValueError:
        mp4_files = []
    combined = dicom_files + mp4_files
    if len(combined) == 0:
        raise ValueError("No DICOM or MP4 files found!")
    return combined


def create_output_directories(outdir: os.PathLike, save_vectors: bool, save_plots: bool):
    """
    Create the output directory (or directories).

    Parameters:
        outdir (os.PathLike):  Output directory.
        save_vectors (bool):   Flag to save vectors.
        save_plots (bool):     Flag to save plots.
    """
    os.makedirs(outdir, exist_ok=True)
    # Create the vectors subdirectory, if saving vectors
    if save_vectors:
        vector_dir = os.path.join(outdir, 'vectors')
        os.makedirs(vector_dir, exist_ok=True)
    # Create the plots subdirectory, if saving plots
    if save_plots:
        plots_dir = os.path.join(outdir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)


def save_vectors_dict(output_dir: os.PathLike,
                      sample_id: os.PathLike,
                      vectors_dict: Dict[str, torch.Tensor]) -> None:
    """
    This function saves each vector from the vectors_dict to a file.
    The file is saved in the 'vectors' subdirectory of the output_dir
    with the name format as '{vector_name}_{uid}.pt' where uid is the
    basename of the input_dicom file.

    Parameters:
        output_dir (os.PathLike):                Directory where the vector files will be saved.
        sample_id (str):               Sample identifier, typically the basename of the input DICOM file.
                                        Determines part of the filename for the saved vectors.
        vectors_dict (Dict[str, torch.Tensor]):  Dictionary where the key is the vector name and
                                                 the value is the vector.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the 'vectors' subdirectory if it doesn't exist
    vectors_dir = os.path.join(output_dir, 'vectors')
    os.makedirs(vectors_dir, exist_ok=True)

    # Write each vector, one at a time.
    for name, vector in vectors_dict.items():
        fname = f"{name}_{sample_id}.pt"
        fpath = os.path.join(vectors_dir, fname)

        # Try to save the feature vector and handle any exceptions
        try:
            torch.save(vector, fpath)
        except Exception as e:
            print(f"Failed to save vector '{fname}' to '{fpath}'. Error: {e}")


def save_intermediate_vectors(output_dir: os.PathLike,
                              sample_id: str,
                              frame_features: torch.Tensor,
                              context_vectors: torch.Tensor,
                              attention_scores: torch.Tensor) -> None:
    """
    This function saves each vector from the vectors_dict to a file. The file is
    saved in the 'vectors' subdirectory of the output_dir with the name format as
    '{vector_name}_{uid}.pt' where uid is the basename of the input_dicom file.

    Parameters:
        output_dir (os.PathLike):          Directory where the vector files will be saved.
        sample_id (str):               Sample identifier, typically the basename of the input DICOM file.
                                        Determines part of the filename for the saved vectors.
        frame_features (torch.Tensor):     Frame features output from CNN
        context_vectors (torch.Tensor):    Context vector(s) from the RNN
        attention_scores (torch.Tensor):   Attention scores from the RNN

    Returns:
        None
    """
    save_vectors_dict(output_dir=output_dir,
                      sample_id=sample_id,
                      vectors_dict={
                          'feature_vector': frame_features,
                          'context_vector': context_vectors,
                          'attention_weights': attention_scores,
                      })


def video_level_inference(model: LightningModule,
                          dicomlist: List[str],
                          outdir: os.PathLike,
                          save_vectors: bool,
                          save_plots: bool,
                          label_plot_name: str,
                          unit_plot_name: str,
                          mode: str,
                          lmean: float = LGA_MEAN,
                          lstd: float = LGA_STD,
                          physicalDeltaX: float = None,
                          ) -> dict:
    """
    Run inference on each set of `frames` individually and record the frame
    feature vectors for running exam-level inference.

    Parameters:
        model (LightningModule):  Model to use for inference.
        dicomlist (List[str]):    List of DICOM files.
        outdir (os.PathLike):     Directory to output the results.
        save_vectors (bool):      Flag to save vectors.
        save_plots (bool):        Flag to save plots.
        label_plot_name (str):    The name of the label visualized in the plots.
        unit_plot_name (str):     The unit of the label visualized in the plots.
        mode (str):               The mode of the model.
        lmean (float):            The mean of the log values for rescaling.
        lstd (float):             The std of the log values for rescaling.
        physicalDeltaX (float):   The physical delta X value from the DICOM metadata, or supplied to the function.

    Returns:
        dict:                     Dictionary containing the results.
    """
    results = defaultdict(list)

    for dicompath in dicomlist:
        # Step 4a. Prepare the frame data from the DICOM for the model
        frames = get_dicom_frames(
            dicompath, device=model.device, mode=mode, pdx=physicalDeltaX)

        # If frames is returned as a `None`, skip this DICOM file
        if frames is None:
            print(
                f"Skipping {dicompath} due to empty frames. Inspect video data for issues.")
            continue

        # Step 4b. Perform inference on the data
        with torch.no_grad():
            # noinspection PyTupleAssignmentBalance
            pred, y_hat, frame_features, context_vectors, attention_scores = \
                run_inference(model, frames, mode=mode, lmean=lmean, lstd=lstd)

        # Step 4c. Record the results and context vector
        results['paths'].append(dicompath)
        if mode.upper() == 'GA':
            results[f'Predicted Log {label_plot_name}'].append(y_hat)
            results[f'Predicted {label_plot_name} ({unit_plot_name})'].append(
                pred)
            pred_to_print = pred
        elif mode.upper() == 'EFW':
            results[f'Predicted Log AC'].append(y_hat[0])
            results[f'Predicted Log FL'].append(y_hat[1])
            results[f'Predicted Log HC'].append(y_hat[2])
            results[f'Predicted Log BPD'].append(y_hat[3])
            results[f'Predicted AC (mm)'].append(pred[0])
            results[f'Predicted FL (mm)'].append(pred[1])
            results[f'Predicted HC (mm)'].append(pred[2])
            results[f'Predicted BPD (mm)'].append(pred[3])
            results[f'Predicted {mode} ({unit_plot_name})'].append(pred[4])
            pred_to_print = pred[4]
        results['frame_features'].append(frame_features)

        # Step 4d. Print final outputs:
        print(f"For input dicom: {dicompath}")
        # noinspection PyUnboundLocalVariable
        print(
            f"Predicted {label_plot_name} ({unit_plot_name}): {pred_to_print}")

        # Step 4e. Save output vectors
        if save_vectors:
            save_intermediate_vectors(outdir,
                                      os.path.basename(dicompath),
                                      frame_features,
                                      context_vectors,
                                      attention_scores)

        # Step 4e. Save attention vector plots
        if save_plots:
            # Get the video_names prepared
            video_name = os.path.basename(dicompath)
            # Plot and save softmax-ed attention
            fig = plot_attention_scores(
                attention_scores, video_name)
            # Check the output directory exists
            os.makedirs(os.path.join(outdir, 'plots'), exist_ok=True)
            # Save the figure
            outpath = os.path.join(outdir, 'plots', f'{video_name}.png')
            fig.savefig(outpath)
            plt.close(fig)

    return results


def video_level_inference_FP(model: LightningModule,
                             dicomlist: List[str],
                             outdir: os.PathLike,
                             save_vectors: bool,
                             lmean: float = LGA_MEAN,
                             lstd: float = LGA_STD,
                             physicalDeltaX: float = None,
                             ) -> dict:
    """
    Run inference on each set of `frames` individually and record the frame
    feature vectors.

    Parameters:
        model (LightningModule):  Model to use for inference.
        dicomlist (List[str]):    List of DICOM files.
        outdir (os.PathLike):     Directory to output the results.
        save_vectors (bool):      Flag to save vectors.
        lmean (float):            The mean of the log values for rescaling.
        lstd (float):             The std of the log values for rescaling.
        physicalDeltaX (float):   The physical delta X value (pixel spacing) from
                                  the DICOM metadata, or supplied to the function.

    Returns:
        dict:                     Dictionary containing the results.
    """
    results = {
        'paths': [],
        'Predicted presentation': [],
        'logits': [],
        'softmax_output': [],
        'frame_features': []
    }

    presentation_dict = {'0': 'cephalic', '1': 'noncephalic'}

    for dicompath in dicomlist:
        # Step 4a. Prepare the frame data from the DICOM for the model
        frames = get_dicom_frames(
            dicompath, model.device, mode='FP', pdx=physicalDeltaX)

        # Step 4b. Perform inference on the data
        with torch.no_grad():
            # noinspection PyTupleAssignmentBalance
            pred, y_hat, frame_features, context_vectors, auxiliary_vectors, logits = \
                run_inference(model, frames, mode='FP', lmean=lmean, lstd=lstd)

        softmax_output = np.exp(logits) / np.sum(np.exp(logits))

        # Step 4c. Print final outputs:
        print(f"For input dicom: {dicompath}")
        print(f"Predicted presentation: {presentation_dict[str(pred[0])]}")

        # Step 4d. Record the results and context vector
        results['paths'].append(dicompath)
        results['Predicted presentation'].append(pred[0])
        results['logits'].append(logits)
        results['softmax_output'].append(softmax_output)
        results['frame_features'].append(frame_features)

        # Step 4e. Save output vectors
        if save_vectors:
            save_intermediate_vectors(outdir,
                                      os.path.basename(dicompath),
                                      frame_features,
                                      context_vectors,
                                      auxiliary_vectors)

    return results


def exam_level_inference(model: LightningModule,
                         results: dict,
                         save_plots: bool,
                         outdir: os.PathLike,
                         label_plot_name: str = "GA",
                         unit_plot_name: str = "Days",
                         mode: str = "GA",
                         lmean: float = LGA_MEAN,
                         lstd: float = LGA_STD,
                         ) -> dict:
    """
    Perform exam-level inference.

    Parameters:
        model (LightningModule):  The model to use for inference.
        results (dict):           Dictionary to store the results.
        save_plots (bool):        Flag to save plots.
        outdir (os.PathLike):     Directory to save results.
        label_plot_name (str):    The name of the label visualized in the plots.
        unit_plot_name (str):     The unit of the label visualized in the plots.
        mode (str):               The mode of the model. GA or EFW.
        lmean (float):            The mean of the log values for rescaling.
        lstd (float):             The std of the log values for rescaling.

    Returns:
        dict:                     Dictionary containing the results.
    """
    # Step 5. Perform exam-level evaluation.
    # If there are more than one video in the dicomlist, treat them as an exam
    # don't forget to save context/attention/frame feature vectors
    # or plots, if requested!

    # Prepare the frame features as torch.Tensor on the correct device.
    # Stacking must be done without batch dimension, which gets added back after vstack.
    all_frame_features = []
    # Also record the video lengths for plotting exam-level attention scores
    video_lengths = []
    # Perform loop
    for x in results['frame_features']:
        # Convert to Tensor and squeeze off the batch dimension.
        x = torch.from_numpy(x)
        x = torch.squeeze(x, dim=0)
        # Send to device and append to the all_frame_features list for stacking.
        x = x.to(model.device)
        all_frame_features.append(x)
        # Compute and record the video length
        video_length = x.shape[0]
        video_lengths.append(video_length)

    # Stack up the feature vectors for all the videos in the exam
    all_frame_features = torch.vstack(all_frame_features)

    # Add batch dimension back
    all_frame_features = torch.unsqueeze(all_frame_features, dim=0)

    # Get the context vector, attention scores, and predicted log GA
    # from the exam
    with torch.no_grad():
        context_vector, attention_scores = model.rnn_forward(
            all_frame_features)
        y_hat = model.regressor(context_vector)

    # Detach the parameters, as before:
    output = [y_hat, context_vector, attention_scores]

    # Detach and convert to numpy all result Tensor values
    output = detach_and_convert_tensors(model, output)

    # Set the exam name from the first video path
    example_path = results['paths'][0]
    exam_name = os.path.dirname(example_path)
    results['paths'].append(f"Exam {exam_name}")

    if mode == 'GA':
        # Create the rescale function for the log GA values
        rescale_log_ga = get_rescale_log_value_func(lmean, lstd)

        # Unpack the values again
        y_hat, context_vector, attention_scores = output

        # Rescale the predicted value, y_hat
        y_hat_days = rescale_log_ga(y_hat)

        # Extract the y_hat and y_hat_days items
        y_hat = y_hat.item()
        y_hat_days = y_hat_days.item()
        # Set the result to be printed
        print_result = y_hat_days
        # Step 4d. Record the results and context vector

        results[f'Predicted Log {label_plot_name}'].append(y_hat)
        results[f'Predicted {label_plot_name} ({unit_plot_name})'].append(
            y_hat_days)
    elif mode == 'EFW':
        # Unpack the values again
        y_hat, context_vector, attention_scores = output
        # Rescale the y_hat values to EFW biometric values
        log_ac, log_fl, log_hc, log_bpd = y_hat[0]
        # Each biometric is rescaled individually using the BioNormLogRescale class
        ac_pred = rescale_log_value(log_ac, AC_MEAN, AC_STD)
        fl_pred = rescale_log_value(log_fl, FL_MEAN, FL_STD)
        hc_pred = rescale_log_value(log_hc, HC_MEAN, HC_STD)
        bpd_pred = rescale_log_value(log_bpd, BPD_MEAN, BPD_STD)
        efw_pred = efw_hadlock_4component(bpd_pred, ac_pred, hc_pred, fl_pred)

        # Set the result to be printed
        print_result = efw_pred
        # Save the results to the results dictionary
        results[f'Predicted Log AC'].append(log_ac.item())
        results[f'Predicted Log FL'].append(log_fl.item())
        results[f'Predicted Log HC'].append(log_hc.item())
        results[f'Predicted Log BPD'].append(log_bpd.item())
        results[f'Predicted AC (mm)'].append(ac_pred.item())
        results[f'Predicted FL (mm)'].append(fl_pred.item())
        results[f'Predicted HC (mm)'].append(hc_pred.item())
        results[f'Predicted BPD (mm)'].append(bpd_pred.item())
        results[f'Predicted {mode} ({unit_plot_name})'].append(efw_pred.item())
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    # Step 5. Print final outputs:
    print("\n")
    print(f"Exam Level Prediction for {exam_name}")
    print(f"Predicted {label_plot_name}  ({unit_plot_name}): {print_result}")

    # Step 6. Save attention vector plots
    if save_plots:
        # Get the video_names prepared
        video_names = [os.path.basename(x) for x in results['paths']]
        # Plot and save softmax-ed attention scores
        fig = plot_exam_attention_scores(
            attention_scores, video_lengths, video_names)
        # Save the figure
        outpath = os.path.join(outdir, 'plots', 'exam_attention.png')
        fig.savefig(outpath)

    return results


def exam_level_inference_FP(results: dict,
                            logits_avg: bool = False):
    """
    Perform exam-level inference.

    Parameters:.
        results (dict):           Dictionary that stores the results.
        logits_avg (bool):        Flag to average the logits before softmax.

    Returns:
        dict:                     Dictionary containing the results.
    """

    presentation_dict = {'0': 'cephalic', '1': 'noncephalic'}

    logits = results['logits']
    # Average the logits as opposed to the softmax values
    if logits_avg:
        logits = np.vstack(logits)
        exam_logits = np.mean(logits, axis=0)
        exam_probs = np.exp(exam_logits) / np.sum(np.exp(exam_logits))
    # Otherwise, softmax the logits and then average
    else:
        logits = np.vstack(logits)
        # Placeholder for now
        exam_logits = None
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        exam_probs = np.mean(probs, axis=0)

    exam_label = np.argmax(exam_probs)
    presentation = presentation_dict[str(exam_label)]

    # Step 4c. Print final outputs:
    example_path = results['paths'][0]
    exam_name = os.path.dirname(example_path)
    print("\n")
    print(f"Exam Level Prediction for {exam_name}")
    print(f"Predicted presentation: {presentation}")

    # Step 4d. Record the results and context vector
    results['paths'].append(f"Exam {exam_name}")
    results['Predicted presentation'].append(exam_label)
    results['logits'].append(exam_logits)
    results['softmax_output'].append(exam_probs)

    return results


def write_results(results: dict,
                  outdir: os.PathLike) -> None:
    """
    Converts a dictionary in a Pandas DataFrame-ready format to a DataFrame,
    then saves it to the output directory as 'results.csv' without the index
    column.

    Parameters:
        results: (dict)        Dictionary in a Pandas DataFrame-ready format.
        outdir: (os.PathLike)  Output directory for the results.

    Returns
        None
    """
    outpath = os.path.join(outdir, 'results.csv')
    df = pd.DataFrame(data=results)
    df.to_csv(outpath, index=False)


def find_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """
    This function searches for an instance of WandbLogger in the loggers of the given trainer.

    Parameters:
        trainer (Trainer): The trainer object which contains the loggers.

    Returns:
        WandbLogger: The first WandbLogger instance found in the trainer's loggers. 
                     Returns None if no WandbLogger instance is found.
    """
    logger = None

    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            break

    return logger


def yaml_to_dict(file_path: str) -> Any:
    """
    This function reads a YAML file and converts it to a Python dictionary.

    Parameters:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The Python dictionary representation of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
