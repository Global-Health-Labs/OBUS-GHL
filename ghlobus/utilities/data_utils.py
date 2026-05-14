"""
data_utils.py

Various utilities related to data ingestion and pre-processing.
Some utilities are designed for a Pytorch `torch.Dataset` to use
for processing data as input to a model.

They can be used standalone if the parameters are specified.

Author: Daniel Shea
        Courosh Mehanian
        Sourabh Kulhare
        Olivia Zahn

This software is licensed under the MIT license. See LICENSE.txt in the root of
the repository for details.
"""
import os
import cv2
import imageio

from tqdm import tqdm
from typing import Callable, List, Union, Tuple

import torch
import random
import pydicom
import numpy as np
import pandas as pd

from torchvision import transforms
from pydicom.tag import Tag
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom import dcmread, config

from ghlobus.utilities.expt_config import centercrop
from ghlobus.utilities.sweep_utils import get_tag_selection
from ghlobus.utilities.constants import BAD_SHAPE
from ghlobus.utilities.constants import BAD_BBOX
from ghlobus.utilities.constants import GOOD_VIDEO_MSG
from ghlobus.utilities.constants import DICOM_FILE_TYPE
from ghlobus.utilities.constants import MP4_FILE_TYPE

config.settings.reading_validation_mode = config.IGNORE


def read_mp4_video(filepath: str,
                   tag: str,
                   raw_frame_index: int = 5,
                   doppler_rgb_thresh: int = 10,
                   doppler_pixel_thresh: int = 1000,
                   min_frames: int = 50,
                   max_frames: int = 10000,
                   ) \
        -> Tuple[Union[torch.Tensor, None],
                 Union[str, None],
                 tuple,
                 Union[np.ndarray, None],
                 ]:
    """
    Read an MP4 video file and return the frames as a Tensor.

    Process as follows:
       dims   |   channels   |   take mean of channels   |   expand dims
    ------------------------------------------------------------------------
        4     |       3      |            yes            |      no
        4     |       1      |            no             |      no
        3     |       1      |            no             |      yes

    Parameters:
        filepath: str                   Path to the MP4 video file
        tag: str                        Sweep tag, e.g., BPD, M, L0, etc.,
                                            used to gate doppler filtering
        raw_frame_index: int            Index of the frame to extract from the video
        doppler_rgb_thresh: int         Threshold for gray level Doppler filtering
        doppler_pixel_thresh: int       Threshold for pixel count Doppler filtering
        min_frames: int                 Minimum number of frames in a video
        max_frames: int                 Maximum MP4 frames to read before rejecting

    Returns:
        frames: torch.Tensor            Frames of the video as a Tensor (or None)
        err_msg: str                    Error message if the video could not be read (or None)
        shape: tuple                    Shape of the frames Tensor
        frame5: np.ndarray              Frame #RAW_FRAME_INDEX of video as ndarray (or None)
    """
    # read the MP4 video file
    try:
        reader = imageio.get_reader(filepath, format='ffmpeg')
    except:
        return None, "Read_MP4_failed", BAD_SHAPE, None

    try:
        nframes = reader.count_frames()
        if nframes > max_frames:
            reader.close()
            print(
                f"Rejecting MP4 with excessive frame count: "
                f"{filepath} has {nframes} frames, max allowed is {max_frames}."
            )
            return None, "Excessive_MP4_frames", (nframes, -99, -99, -99), None
    except:
        pass

    # step through frame-by-frame
    try:
        frames = list()
        for idx, im in enumerate(reader):
            if idx >= max_frames:
                reader.close()
                print(
                    f"Rejecting MP4 after reading too many frames: "
                    f"{filepath} exceeded {max_frames} frames."
                )
                return None, "Excessive_MP4_frames", (idx + 1, -99, -99, -99), None
            frames.append(im)
        reader.close()
    except MemoryError:
        # close the video capture
        reader.close()
        return None, "MP4_memory_error", BAD_SHAPE, None
    except:
        # close the video capture
        reader.close()
        return None, "Frame_step_MP4_failed", BAD_SHAPE, None

    # convert to numpy array
    try:
        frames = np.array(frames)
    except:
        return None, "Convert_array_MP4_failed", BAD_SHAPE, None

    # Extract #RAW_FRAME_INDEX frame
    if frames.shape[0] > 0:
        frame5 = frames[min(raw_frame_index, frames.shape[0] - 1)]
    else:
        return None, "Invalid_MP4_shape", BAD_SHAPE, None

    try:
        # find number of dimensions
        dims = len(frames.shape)

        # expand to 4 dims if needed
        if dims == 3:
            frames = np.expand_dims(frames, axis=-1)

        # get original shape
        shape = frames.shape

        # find number of channels
        num_channels = shape[-1]

        # Swap axes so the dimensions are (n,c,h,w) instead of (n,h,w,c)
        shape = tuple(np.array(shape)[[0, 3, 1, 2]])

        # find number of frames
        num_frames = shape[0]

        if num_frames < min_frames:
            return None, "Insufficient_MP4_frames", shape, frame5

        # check if the video is color Doppler
        if num_channels == 3:
            if check_doppler(frame5,
                             tag,
                             gray_threshold=doppler_rgb_thresh,
                             pixel_threshold=doppler_pixel_thresh,
                             ):
                return None, "Doppler_RGB_MP4", shape, frame5

            # take mean of the three channels
            frames = frames.mean(axis=-1, keepdims=True)

        # convert to Tensor
        frames = torch.Tensor(frames).byte()

        # Swap axes so the dimensions are (n,c,h,w) instead of (n,h,w,c)
        frames = frames.permute(0, 3, 1, 2)
    except:
        return None, "Process_MP4_failed", BAD_SHAPE, frame5

    return frames, None, shape, frame5


def preprocess_video(file_info: pd.Series,
                     out_dir: Union[str, None] = None,
                     img_size: int = 288,
                     channels: int = 1,
                     dtype: str = 'uint8',
                     alpha: float = 0.075,
                     min_frames: int = 50,
                     max_mp4_frames: int = 10000,
                     sample_frames: Union[int, None] = None,
                     doppler_rgb_thresh: int = 10,
                     doppler_ybr_thresh: int = 100,
                     doppler_pixel_thresh: int = 1000,
                     raw_frame_index: int = 5,
                     jpeg_quality: int = 100,
                     raw: Union[str, None] = None,
                     jpg: Union[str, None] = None,
                     pt: Union[str, None] = None,
                     force_dcm_load: bool = False,
                     ) -> Union[pd.DataFrame, torch.Tensor, None]:
    """
    Process the dicom file path and write output video clips or return video frames.

    Parameters:
        file_info: pd.Series        DataFrame row containing the following columns
                                        in_filepath:  input file path
                                        file_type:    file type
                                        project:      project folder name
                                        exam_dir:     output exam folder name (not full path)
                                        tag:          sweep tag, e.g., BPD, M, L0, etc.
                                        pdx:          PhysicalDeltaX (MP4 only)
        out_dir: str                Root directory for output data
        img_size: int               Size of output image in # of pixels
        channels: int               Determines channel count according to:
                                    When the input is three-channel and channels=1:
                                        take mean of three channels
                                    When the input is single-channel and channels=3:
                                        duplicate the single-channel image to 3 channels
        dtype: str                  'uint8' or 'float32'
        alpha: float                Desired final alpha when images are scaled (resized, resampled)
        min_frames: int             Minimum number of frames in a video
        max_mp4_frames: int         Maximum MP4 frames to read before rejecting
        sample_frames: int          Number of frames in a video to sample and write as jpgs
                                        set to None to write all frames
        doppler_rgb_thresh: int     Threshold for gray level Doppler filtering
        doppler_ybr_thresh: int     Threshold for YBR Doppler filtering
        doppler_pixel_thresh: int   Threshold for pixel count Doppler filtering
        raw_frame_index: int        Index of the frame to extract from the video
        jpeg_quality: int           JPEG quality for writing frames as JPEG
        raw: str or None            Top-level folder to write raw 5th frame #RAW_FRAME_INDEX
                                        of every video as PNG
        jpg: str or None            Top-level folder to write post-processed frames as JPEG
        pt: str or None             Top-level folder to write output pytorch video data to disk
                                        if None, just return video frames
                                        NOTE: if both jpg and pt are None
                                        post-processed frames are returned but not saved to disk
        force_dcm_load: bool        flag that causes dcmread to load DICOM file even when
                                        File Meta Information header is missing

    Returns:
        frames: torch.Tensor or pandas.DataFrame   pixel_data in the DICOM file in
                                                   RGB PhotometricInterpretation/encoding
    """
    # Unpack file_info, common to dicom and MP4 videos
    in_filepath = file_info['in_filepath']
    ftype = file_info['file_type']
    exam_dir = file_info['exam_dir']
    project = file_info['project']
    tag = file_info['tag']
    pdx = file_info['pdx']

    # Check if we are writing anything to disk
    if raw is None and jpg is None and pt is None:
        # Not writing anything to disk, just outputting frames
        write = False
        file_base_name = None
    else:
        # Writing at least something to disk
        write = True
        # Construct the base file name for writing output files
        file_base_name = os.path.basename(in_filepath)
        # Determine if there is an extension in the filename
        if file_base_name[-3:] == DICOM_FILE_TYPE or \
                file_base_name[-3:] == MP4_FILE_TYPE:
            file_base_name = os.path.splitext(file_base_name)[0]

    # Look for the input file
    if not os.path.isfile(in_filepath):
        if write:
            # Return stats and reason for failure
            return create_video_df("File_not_found",
                                   BAD_SHAPE,
                                   BAD_SHAPE,
                                   BAD_BBOX)
        else:
            return None

    # Input file exists, look for output file
    if pt is not None:
        # For ingestion_v4, pt will be "", a blank string (not None)
        pt_path = os.path.join(out_dir,
                               pt,
                               project,
                               exam_dir,
                               file_base_name + '.pt')

    # Image size as tuple
    final_dims = (img_size,) * 2

    # Determine file type
    if ftype == DICOM_FILE_TYPE:
        # Load the DICOM file
        try:
            with open(in_filepath, "rb") as f:
                dcm = dcmread(f, force=force_dcm_load)
        except:
            if write:
                # Return stats and reason for failure
                return create_video_df("Dicom_read_failed",
                                       BAD_SHAPE,
                                       BAD_SHAPE,
                                       BAD_BBOX)
            else:
                return None

        # See if it is a valid blind sweep DICOM file
        err_msg, bbox = check_dicom(dcm)
        if err_msg is not None:
            if write:
                # Return stats and reason for failure
                return create_video_df(err_msg,
                                       BAD_SHAPE,
                                       BAD_SHAPE,
                                       bbox)
            else:
                return None

        # Extract frames as RGB frames
        frames, err_msg, shape_in, frame5 = extract_rgb_frames(
            dcm,
            tag,
            channels=channels,
            dtype=dtype,
            raw_frame_index=raw_frame_index,
            doppler_rgb_thresh=doppler_rgb_thresh,
            doppler_ybr_thresh=doppler_ybr_thresh,
            doppler_pixel_thresh=doppler_pixel_thresh,
            min_frames=min_frames,
            return_status=True)

        # Something went wrong with frame extraction
        if frames is None:
            if write:
                # Write #RAW_FRAME_INDEX frame of the video as PNG?
                if raw is not None and frame5 is not None:
                    write_frame5(frame5,
                                 out_dir,
                                 raw,
                                 project,
                                 exam_dir,
                                 file_base_name,
                                 reason=fail_reason_acronym(err_msg))
                # Return stats and reason for failure
                return create_video_df(err_msg,
                                       shape_in,
                                       BAD_SHAPE,
                                       bbox)
            else:
                return None

        # Crop and rescale the frames with the target alpha
        frames = crop_and_scale_frames_tensor(frames,
                                              dcm,
                                              alpha,
                                              bbox=bbox,
                                              return_on_fail=False)

        if frames is None:
            if write:
                err_msg = "Resize_failed_DICOM"
                # Write #RAW_FRAME_INDEX frame of the video as PNG?
                if raw is not None and frame5 is not None:
                    write_frame5(frame5,
                                 out_dir,
                                 raw,
                                 project,
                                 exam_dir,
                                 file_base_name,
                                 reason=fail_reason_acronym(err_msg))
                # Return stats and reason for failure
                return create_video_df(err_msg,
                                       shape_in,
                                       BAD_SHAPE,
                                       bbox)
            else:
                return None

        # Explicitly delete dicom from memory
        del dcm

    elif ftype == MP4_FILE_TYPE:
        # For MP4 video files, there is no bounding box
        bbox = BAD_BBOX

        # Load the MP4 video file
        frames, err_msg, shape_in, frame5 = read_mp4_video(
            in_filepath,
            tag,
            raw_frame_index=raw_frame_index,
            doppler_rgb_thresh=doppler_rgb_thresh,
            doppler_pixel_thresh=doppler_pixel_thresh,
            min_frames=min_frames,
            max_frames=max_mp4_frames,
        )

        if frames is None:
            if write:
                # Write #RAW_FRAME_INDEX frame of the video as PNG?
                if raw is not None and frame5 is not None:
                    write_frame5(frame5,
                                 out_dir,
                                 raw,
                                 project,
                                 exam_dir,
                                 file_base_name,
                                 reason=fail_reason_acronym(err_msg))
                # Return stats and reason for failure
                return create_video_df(err_msg,
                                       shape_in,
                                       BAD_SHAPE,
                                       bbox)
            else:
                return None

        # Get the physical size scaling factor
        try:
            physicalDeltaX = float(pdx)
        except:
            if write:
                err_msg = "No_PhysicalDeltaX_MP4"
                if raw is not None and frame5 is not None:
                    write_frame5(frame5,
                                 out_dir,
                                 raw,
                                 project,
                                 exam_dir,
                                 file_base_name,
                                 reason=fail_reason_acronym(err_msg))
                # Return stats and reason for failure
                return create_video_df(err_msg,
                                       shape_in,
                                       BAD_SHAPE,
                                       bbox)
            else:
                return None

        # Crop and rescale the frames with the target alpha
        frames = crop_and_scale_frames_tensor(frames,
                                              physicalDeltaX,
                                              alpha,
                                              bbox=None,
                                              return_on_fail=False)

        if frames is None:
            if write:
                err_msg = "Resize_failed_MP4"
                # Write #RAW_FRAME_INDEX frame of the video as PNG?
                if raw is not None and frame5 is not None:
                    write_frame5(frame5,
                                 out_dir,
                                 raw,
                                 project,
                                 exam_dir,
                                 file_base_name,
                                 reason=fail_reason_acronym(err_msg))
                # Return stats and reason for failure
                return create_video_df(err_msg,
                                       shape_in,
                                       BAD_SHAPE,
                                       bbox)
            else:
                return None

    else:
        if write:
            # Return stats and reason for failure
            return create_video_df(f"Invalid_ftype_{ftype}",
                                   BAD_SHAPE,
                                   BAD_SHAPE,
                                   BAD_BBOX)
        else:
            return None

    # Apply padding to the frames
    frames = pad_to_dimensions(frames, target_dims=final_dims)

    # Check whether we are saving to disk
    if write:
        # Use case: saving to disk, i.e., preprocessing pipeline
        # Parse the various write folders (directives: raw, jpg, pt)

        # Write #RAW_FRAME_INDEX frame of the video as PNG?
        if raw is not None and frame5 is not None:
            write_frame5(frame5,
                         out_dir,
                         raw,
                         project,
                         exam_dir,
                         file_base_name,
                         reason=None)

        # Write individual frames of the video as JPGs?
        if jpg is not None and frames is not None:
            jpg_path = os.path.join(
                out_dir, jpg, project, exam_dir, file_base_name)
            numpy_frames = frames.permute(0, 2, 3, 1).cpu().numpy()
            # Generate random sample or take all
            if sample_frames is None:
                # take all
                frame_inds = list(range(numpy_frames.shape[0]))
            else:
                # Random sample
                frame_inds = sorted(random.sample(
                    range(numpy_frames.shape[0]), k=sample_frames))
            # Assemble (sampled) frames
            numpy_frames = numpy_frames[frame_inds]
            for index, frame in zip(frame_inds, numpy_frames):
                frame_path = jpg_path + f"_#{index:04d}.jpg"
                # Write the frame; first check if file exists
                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, frame, [
                                cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Write whole video as PT?
        if pt is not None and frames is not None:
            # Write the video as pytorch file
            # Don't write file if it already exists
            # noinspection PyUnboundLocalVariable
            if not os.path.exists(pt_path):
                torch.save(frames, pt_path)

        # Frame shape:
        shape_out = tuple(frames.shape)

        # Explicitly delete frames Tensor from memory
        del frames

        # Create dataframe with file information
        video_df = create_video_df(GOOD_VIDEO_MSG,
                                   shape_in,
                                   shape_out,
                                   bbox)

        # Return shape information as pandas DataFrame
        return video_df
    else:
        # Use case: processing single video at a time
        return frames


def write_frame5(frame5: np.ndarray,
                 out_dir: str,
                 raw: str,
                 project: str,
                 exam_dir: str,
                 file_base_name: str,
                 reason: Union[None, str] = None):
    """
    Write the 5th frame of the video as a PNG file.

    Parameters
        frame5: np.ndarray      Frame #RAW_FRAME_INDEX of video as ndarray
        out_dir: str            root directory for output data
        raw: str                name of folder to write raw 5th frame
                                    #RAW_FRAME_INDEX as PNG
        project: str            project name (for creating output data folder name)
        exam_dir: str           output exam folder name (not full path)
        file_base_name: str     base name of the file (without extension)
        reason: str or None     reason video is good (None) or bad

    Returns
        None
    """
    # Construct the path to write the frame
    if reason is None:
        raw_path = os.path.join(out_dir,
                                raw,
                                project,
                                f"{exam_dir}_{file_base_name}_#0005.png")
    else:
        raw_path = os.path.join(out_dir,
                                raw,
                                project + "_bad",
                                f"{exam_dir}_{file_base_name}_{reason}_#0005.png")

    # Write the frame as a PNG file
    # Don't write file if it already exists
    if not os.path.exists(raw_path):
        cv2.imwrite(raw_path, frame5)


def fail_reason_acronym(reason: str) -> str:
    """
    Compresses failure reason to 3-letter acronym

    Parameters
        reason: str   Reason for failure

    Returns:
        str:          3-letter acronym for failure reasons
    """
    code_dict = {
        "File_not_found": 'FNF',
        "PT_file_exists": 'PTE',
        "Dicom_read_failed": "DRF",
        "Invalid_ultrasound_region_dicom": "URD",
        "Invalid_bounding_box_dicom": "BBD",
        "MONOCHROME_shape_invalid": "MSD",
        "Multichannel_shape_invalid": "CSD",
        "Insufficient_YBR_dicom_frames": "YFD",
        "Doppler_YBR_dicom": "DYD",
        "Insufficient_MONO_dicom_frames": "MFD",
        "Insufficient_RGB_dicom_frames": "RFD",
        "Doppler_RGB_dicom": "DRD",
        "Resize_failed_DICOM": "ZFD",
        "Read_MP4_failed": "RFM",
        "Invalid_MP4_shape": "ISM",
        "Insufficient_MP4_frames": "IFM",
        "Doppler_RGB_MP4": "DRM",
        "Process_MP4_failed": "PFM",
        "No_PhysicalDeltaX_MP4": "PXM",
        "Resize_failed_MP4": "ZFM",
    }

    return code_dict[reason]


def extract_rgb_frames(
        dcm: pydicom.dataset.FileDataset,
        tag: str,
        channels: int = 3,
        dtype: str = 'uint8',
        raw_frame_index: int = 5,
        doppler_rgb_thresh: int = 10,
        doppler_ybr_thresh: int = 100,
        doppler_pixel_thresh: int = 1000,
        min_frames: int = 50,
        return_status: bool = False) \
        -> Union[
            None,
            torch.Tensor,
            Tuple[
                Union[torch.Tensor, None],
                Union[str, None],
                tuple,
                Union[np.ndarray, None],
            ]]:
    """
    Extract the frames in RGB format from the DICOM file.

    Parameters:
        dcm: pydicom.dataset.FileDataset    pyDICOM file, pydicom.dataset.FileDataset,
                                            containing a sequence with crop box information
                                            about the active area of an ultrasound image/video
        tag: str                            Sweep tag, e.g., BPD, M, L0, etc.,
                                                used to gate doppler filtering
        channels: int                       determines channel count according to:
                                                When the input is three-channel and channels=1:
                                                    take mean of three channels
                                                When the input is single-channel and channels=3:
                                                    duplicate the single-channel image to 3 channels
        dtype: str                          'uint8' or 'float32'
        raw_frame_index: int                Index of the frame to extract from the video
        doppler_rgb_thresh: int             Threshold for gray level Doppler filtering
        doppler_ybr_thresh: int             Threshold for YBR Doppler filtering
        doppler_pixel_thresh: int           Threshold for pixel count Doppler filtering
        min_frames: int                     Minimum number of frames in a video
        return_status: bool                 whether to return status or not

    Returns:
        frames: torch.Tensor                pixel_data in the DICOM file, in RGB
                                                PhotometricInterpretation/encoding
                                            First, check the dimensionality of the 'frames' Tensor
                                            It's either single-channel or three channel. (or None)
        err_msg: str                        Error message if the video could not be read (or None)
        shape: tuple                        Shape of the frames Tensor
        frame5: np.ndarray                  Frame #RAW_FRAME_INDEX of video as ndarray
    """
    # Grab the frames as np.ndarray
    frames = dcm.pixel_array

    # get dimensions and shape
    dims = len(frames.shape)

    # Get dimensions of the frames
    if dims == 3:
        shape = frames.shape + (1,)
    else:
        shape = frames.shape

    # permute the shape to (n,c,h,w) instead of (n,h,w,c)
    shape = tuple(np.array(shape)[[0, 3, 1, 2]])

    # find number of frames
    num_frames = shape[0]

    # Get photometric interpretation
    pmi = dcm.PhotometricInterpretation

    # Make sure it's a video and not just an image
    if pmi == 'MONOCHROME2' and dims != 3:
        if return_status:
            return None, "MONOCHROME_shape_invalid", shape, None
        else:
            return None

    elif pmi in ['YBR_FULL', 'RGB'] and dims != 4:
        if return_status:
            return None, "Multichannel_shape_invalid", shape, None
        else:
            return None

    # depending on the photometric interpretation,
    if pmi == 'YBR_FULL':

        # convert the pixel values:
        frames = convert_color_space(frames, 'YBR_FULL_422', 'RGB')
        # for i in range(frames.shape[0]):
        #     frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_YCrCb2RGB)

        # Grab frame #RAW_FRAME_INDEX
        frame5 = frames[min(raw_frame_index, frames.shape[0] - 1)]

        if num_frames < min_frames:
            return None, "Insufficient_YBR_dicom_frames", shape, frame5

        # check if the video is color Doppler
        if check_doppler(frame5,
                         tag,
                         gray_threshold=doppler_ybr_thresh,
                         pixel_threshold=doppler_pixel_thresh,
                         ):
            return None, "Doppler_YBR_dicom", shape, frame5

        # if single-channel requested, average over color channels
        if channels == 1:
            # keep the averaged colors as singleton dimension
            frames = frames.mean(axis=-1, keepdims=True)

        # convert to Torch tensor
        if dtype == 'uint8':
            frames = torch.Tensor(frames).byte()
        else:
            frames = torch.Tensor(frames)

        # Swap axes so the dimensions are (n,c,h,w) instead of (n,h,w,c)
        frames = frames.permute(0, 3, 1, 2)

    elif pmi == 'MONOCHROME2':

        # Grab frame #RAW_FRAME_INDEX
        frame5 = frames[min(raw_frame_index, frames.shape[0] - 1)]

        # adjust the dimensions - now it should be (n,c,h,w)
        if channels == 3:
            # with H,W monochrome scale copied across the channels
            frames = np.repeat(frames[:, None, ...], 3, axis=1)
        else:
            # insert channel singleton dimension of 1
            frames = np.expand_dims(frames, axis=1)

        if num_frames < min_frames:
            return None, "Insufficient_MONO_dicom_frames", shape, frame5

        # convert to Torch tensor
        if dtype == 'uint8':
            frames = torch.Tensor(frames).byte()
        else:
            frames = torch.Tensor(frames)

    elif pmi == 'RGB':

        # Grab frame #RAW_FRAME_INDEX
        frame5 = frames[min(raw_frame_index, frames.shape[0] - 1)]

        if num_frames < min_frames:
            return None, "Insufficient_RGB_dicom_frames", shape, frame5

        # check if the video is color Doppler
        if check_doppler(frame5,
                         tag,
                         gray_threshold=doppler_rgb_thresh,
                         pixel_threshold=doppler_pixel_thresh,
                         ):
            return None, "Doppler_RGB_dicom", shape, frame5

        # if single-channel requested, average over color channels
        if channels == 1:
            # keep the averaged colors as singleton dimension
            frames = frames.mean(axis=-1, keepdims=True)

        # convert to Torch tensor
        if dtype == 'uint8':
            frames = torch.Tensor(frames).byte()
        else:
            frames = torch.Tensor(frames)

        # Swap axes so the dimensions are (n,c,h,w) instead of (n,h,w,c)
        frames = frames.permute(0, 3, 1, 2)

    # Return frames, err_msg, and shape
    if return_status:
        # noinspection PyUnboundLocalVariable
        return frames, None, shape, frame5
    else:
        return frames


def crop_and_scale_frames_tensor(frames: torch.Tensor,
                                 dcm: Union[pydicom.dataset.FileDataset, float],
                                 target_alpha: float,
                                 bbox: Union[tuple, None] = None,
                                 return_on_fail: bool = True) \
        -> Union[torch.Tensor, None]:
    """
    Crop the active region of the pixel_data array,
    then scale the frame data to the targeted size in mm/pixel.

    Parameters:
        frames: torch.Tensor                Frame data to operate on.
        dcm: pydicom.dataset.FileDataset    pyDICOM file containing a sequence with
                                                crop box information or float about the
                                                active area of an ultrasound image/video,
                                                or the physical size scaling factor in
                                                cm/px for an MP4 video.
        target_alpha: float                 target size scale for the frames data, in mm/pixel.
        bbox: tuple                         Bounding box dimensions of ultrasound region from
                                                a previous dicom read, or None if unknown
                                                (process_dicom use case); ignored if MP4.
        return_on_fail: bool                flag to return uncropped frames or None if the
                                                resize operation fails.

    Returns:
        frames: torch.Tensor or None        cropped frames that have been rescaled using Bilinear
                                                interpolation to target size scale.
    """
    # If dcm is float, this is MP4 and dcm is PhysicalDeltaX
    if isinstance(dcm, float):
        # It's an MP4 file
        dicom = False
        # actual scale of video
        alpha = dcm
    else:
        # It's a DICOM file
        dicom = True
        # Extract the cropping bounding box
        if bbox is None:
            x0, y0, x1, y1 = get_crop_box(dcm)
        else:
            x0, y0, x1, y1 = bbox
        # Crop all frames
        frames = frames[..., y0:y1, x0:x1]
        # Extract the existing physical size scale
        alpha = get_physical_scale(dcm)

    # Compute the ratio (existing_alpha)/(target_alpha)
    alpha_ratio = alpha/target_alpha

    # Round off the final image dimensions:
    h, w = frames.shape[-2], frames.shape[-1]
    final_dims = tuple([int(round(alpha_ratio * x, 0)) for x in (h, w)])

    # Set resampling method
    resample_method = transforms.InterpolationMode.BILINEAR
    # Attempt resize
    try:
        # check this transform on RGB 3-channel image(s) with dims ([n,]c,h,w)
        tfx = transforms.Resize(final_dims, resample_method, antialias=True)
        frames = tfx(frames)
    except:
        # resize failed, check if we should return frames anyway or None
        if return_on_fail:
            # for dicom, print message
            if dicom:
                print("Dicom resize failed.")
                print(dcm.Rows, dcm.Columns, dcm.SamplesPerPixel,
                      dcm.StudyInstanceUID,
                      dcm[0x008, 0x0070].value,  # Manufacturer
                      dcm[0x0008, 0x1090].value  # Probe
                      )
            return frames
        else:
            # if return_on_fail is False, return None
            return None

    # resize succeeded
    return frames


def pad_to_dimensions(frames: torch.Tensor, target_dims: tuple) -> torch.Tensor:
    """
    Pad the input torch.Tensor 'frames' to target dimensions in target_dims.

    Parameters:
        frames: torch.Tensor        Frame data to operate on.
        target_dims: tuple[float]   Containing the (H,W) representing the final target
                                    dimensions of the frames Tensor, after padding.

    Returns:
        frames: torch.Tensor        cropped frames that have padded to the target dimensions
    """
    # Extract the current dimensions
    h, w = frames.shape[-2], frames.shape[-1]

    # compute total padding needed in each dimension
    total_pad_y = (target_dims[0] - h)
    total_pad_x = (target_dims[1] - w)

    # Establish padding values; add odd 1 to x0 or y0 if needed
    pad_x0 = int(total_pad_x / 2) + total_pad_x % 2
    pad_x1 = int(total_pad_x / 2)
    pad_y0 = int(total_pad_y / 2) + total_pad_y % 2
    pad_y1 = int(total_pad_y / 2)

    # Set padding for transformation
    # format in torch.nn.functional.pad is (padding_left,padding_right,padding_top,padding_bottom)
    padding = (pad_x0, pad_x1, pad_y0, pad_y1)
    frames = torch.nn.functional.pad(input=frames,
                                     pad=padding,
                                     mode='constant',
                                     value=0)

    # return the transformed frames
    return frames


def prepare_frames(frames: torch.Tensor,
                   channels: int = 3,
                   crop: Callable = centercrop,
                   subsample_method: Callable = None,
                   transforms: List[Callable] = None,
                   frames_or_channel_first='frames',
                   **kwargs,
                   ) -> torch.Tensor:
    """
    Prepares video frames for downstream model usage.

    Args:
        frames (torch.Tensor): A tensor containing video frames.
        channels (int, optional): Number of output channels (default is 3).
        crop (Callable, optional): A cropping function to apply to frames (default is CenterCrop).
        subsample_method (optional): A subsampling method to apply (e.g., temporal downsampling).
        transforms (list, optional): List of additional transformations to apply (default is []).
        frames_or_channel_first (str, optional): Order of frames or channels ('frames' or 'channel').

    Returns:
        torch.Tensor: Processed video frames.
    """

    # Apply the subsampling method
    if subsample_method:
        try:
            frames = subsample_method(frames, **kwargs)
        except Exception as err:
            print("Failed subsample.")
            raise err

    # check if we need to expand channel dimension
    frames_channels = frames.shape[1]
    if channels == 3 and frames_channels == 1:
        # copy monochrome image across three channels
        frames = torch.repeat_interleave(frames, repeats=3, dim=1)

    # Apply the random crop
    if crop:
        try:
            frames = crop(frames)
        except Exception as err:
            print("Failed crop.")
            raise err

    # Apply the provided transforms:
    if transforms:
        try:
            frames = apply_transforms(frames, transforms)
        except Exception as err:
            print("Failed Applying transforms.")
            raise err

    # Check the order of frames or channel, ex. channel first is needed for swin transformer models
    if frames_or_channel_first == 'channel':
        frames = torch.Tensor.movedim(frames, 1, 0)

    return frames


def apply_transforms(frames: torch.Tensor,
                     transforms: List[Callable] = None):
    """
    Apply the transforms in self.transforms to the input frames.

    Parameters:
        frames: torch.Tensor - data Tensor containing frame data.
        transforms: List[Callable] - a list of Callable functions transforming the data.

    Returns:
        frames: torch.Tensor - data modified by the `transforms` functions.
    """
    # Catch the case of no transforms to be applied.
    if not transforms:
        return frames

    # Try applying each transform individually.
    for transform in transforms:
        try:
            frames = transform(frames)
        except Exception as err:
            print("failed transform:\n {transform}")
            print("with exception:", err)
            raise err
    return frames


def retrieve_dicom_tag(dcm, tag_code: str):
    """
    Retrieves the value of a Tag from a dicom file, via the tag_code.

    Parameters:
        dcm:        pyDICOM file, pydicom.dataset.FileDataset, containing tags
        tag_code:   code for a tag within pyDicom file

    Returns:
        value:      value of the tag at tag_code
    """
    # Convert the tag_code to a pydicom.tag.Tag
    t = Tag(tag_code)
    # And use it to access the desired attribute
    return dcm[t].value


def retrieve_dicom_sequence_tag(dcm, tag_code: str):
    """
    Retrieves the value of a sequence of Tags from a dicom file, via the tag_code.
    By default, always chooses the first Tag in the sequence.

    Parameters:
        dcm:        pyDICOM file, pydicom.dataset.FileDataset, containing tags
        tag_code:   code for a tag within pyDicom file

    Returns:
        value:      value of the tag at tag_code
    """
    t = Tag(tag_code)
    # Get the relevant sequence
    # (the first sequence refers to sonogram area)
    seq_tag_code = "00186011"
    seq = retrieve_dicom_tag(dcm, seq_tag_code)
    # Always the first sequence, per above
    seq = seq[0]
    # Return the tag from the sequence
    return seq[t].value


def get_physical_scale(dcm):
    """
    Extract the physical size scaling factor, in cm/px, from a DICOM file.

    Parameters:
        dcm: pydicom.dataset.FileDataset    pyDICOM file, pydicom.dataset.FileDataset, containing a size scaling factor

    Returns:
        alpha: float                        indicating the number of centimeters per pixel (alpha scaling factor)
    """
    return dcm.SequenceOfUltrasoundRegions[0].PhysicalDeltaX


def check_dicom(dcm: pydicom.dataset.FileDataset) -> \
        Tuple[Union[str, None], tuple]:
    """
    Check if the DICOM has a valid Sequence of Ultrasound Regions.

    Parameters
    ----------
        dcm:   pydicom.dataset.FileDataset  pyDICOM file
                                            should contain a sequence contains crop box information
    Returns
    -------
        err_msg: str or None               Error message if DICOM invalid (or None)
        bbox: tuple                        Bounding box dimensions of ultrasound region
    """
    try:
        # Sequence, first refers to Sonogram region
        dcm.SequenceOfUltrasoundRegions[0]
    except:
        return "Invalid_ultrasound_region_dicom", BAD_BBOX

    # get the bounding box
    try:
        bbox = get_crop_box(dcm)
    except:
        return "Invalid_bounding_box_dicom", BAD_BBOX

    # return no error message and the bounding box
    return None, bbox


def check_doppler(image: np.ndarray,
                  tag: str,
                  gray_threshold: int,
                  pixel_threshold: int) -> bool:
    """
    Check if the video is likely color Doppler by computing
    differences in RGB channels. Compute the maximum range of gray
    levels among RGB channels and check if it is greater than
    gray_threshold gray levels, e.g., 10, for more than
    pixel_threshold pixels, e.g., 1000.

    Only check Doppler if tag is not a Blind sweep or biometric tag.

    Parameters
    ----------
        image:   np.ndarray      video np.ndarray
        tag:     str             tag of the video, used to gate doppler filtering
        gray_threshold: int      threshold for gray levels
        pixel_threshold: int     threshold for number of pixels

    Returns
    -------
        status: bool              True if the video is likely color Doppler.
    """
    # Construct tags that obviate Doppler check
    # Include only blind sweep and biometric tags
    tag_select_dict = {'include_biometric': True,
                       'include_unknown': False}
    valid_tags = get_tag_selection(tag_select_dict)

    # See if we need to check doppler based on tag
    if tag in valid_tags:
        doppler = False
    else:
        # It's not a valid tag; check for Doppler

        # collapse the spatial image dimensions
        pixels = image.reshape(-1, 3)

        # compute the range of gray levels among RGB channels
        ranges = np.ptp(pixels, axis=1)

        # check if the range is greater than 10 for more than 1000 pixels
        num_pixels = np.sum(ranges > gray_threshold)
        doppler = num_pixels > pixel_threshold

    return doppler


def create_video_df(reason: str,
                    shape_in: tuple,
                    shape_out: tuple,
                    bbox: tuple,
                    ) -> pd.DataFrame:
    """
    Create a DataFrame with video stats and error message.

    Parameters:
        reason: str         Error message
        shape_in: Tuple     Shape of the input video
        shape_out: Tuple    Shape of the output video
        bbox: Tuple         Bounding box dimensions of ultrasound region

    Returns:
        df: pd.DataFrame    DataFrame with video stats and error message
    """
    # Create a DataFrame with
    df = pd.DataFrame({
        'orig_frames': [shape_in[0]],
        'orig_channels': [shape_in[1]],
        'orig_height': [shape_in[2]],
        'orig_width': [shape_in[3]],
        'scal_frames': [shape_out[0]],
        'scal_channels': [shape_out[1]],
        'scal_height': [shape_out[2]],
        'scal_width': [shape_out[3]],
        'bbox_x0': [bbox[0]],
        'bbox_y0': [bbox[1]],
        'bbox_x1': [bbox[2]],
        'bbox_y1': [bbox[3]],
        'fail_reason': [reason],
    })

    return df


def get_crop_box(dcm) -> tuple:
    """
    Extract the bounding box to crop the ultrasound active region

    Parameters:
        dcm: pydicom.dataset.FileDataset    pyDICOM file
                                            containing a sequence which contains
                                                crop box information about the
                                            active area of an ultrasound image/video

    Returns:
        bbox: list                          crop box dimensions for PIL.Image.crop function
    """
    # Sequence, first refers to Sonogram region
    seq = dcm.SequenceOfUltrasoundRegions[0]

    # Define bbox DICOM keys
    bbox_values = {
        'x0': seq.RegionLocationMinX0,
        'x1': seq.RegionLocationMaxX1,
        'y0': seq.RegionLocationMinY0,
        'y1': seq.RegionLocationMaxY1,
    }

    # Order for downstream cropping
    bbox = (bbox_values['x0'], bbox_values['y0'],
            bbox_values['x1'], bbox_values['y1'])

    return bbox


def empty_cells(df: pd.DataFrame, col: str = 'tag') -> np.ndarray:
    """
    Shortcut for finding any sort of empty entry in DataFrame column.

    Args:
        df: pd.DataFrame     DataFrame to check
        col: str             Column name

    Returns:
        nd.array             Mask of empty values
    """
    empties = np.logical_or.reduce((df[col].isna(),
                                    df[col].isnull(),
                                    df[col] == ''))
    return empties


def read_spreadsheet(filepath: str,
                     sheet: Union[str, None] = None,
                     rows=None) -> pd.DataFrame:
    """
    Read a spreadsheet file into a pandas DataFrame.

    Args:
        filepath (str): The name of the file to read.
        sheet (str): The name of the sheet to read.
        rows (int): The number of rows to read (if known).

    Returns:
        pd.DataFrame: The data read from the file.
    """
    # Determine the file extension
    _, extension = os.path.splitext(filepath)

    # Read the file based on the extension
    if extension == '.csv':
        if rows is not None:
            chunks = int(rows/1000) + 1
        else:
            chunks = None
        df = pd.concat([x for x in tqdm(pd.read_csv(filepath,
                                                    dtype=str,
                                                    keep_default_na=False,
                                                    chunksize=1000),
                                        desc=f"Reading {filepath}",
                                        total=chunks)])
        return df
    elif extension == '.xlsx':
        try:
            return pd.read_excel(filepath, sheet_name=sheet, dtype=str, keep_default_na=False)
        except:
            raise ValueError("Valid sheet not specified for CRF Excel file.")
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


def read_spreadsheet_columns(filepath: str,
                             sheet: Union[str, None] = None,
                             rows: Union[int, None] = None,
                             columns: Union[List[str], None] = None,
                             ) -> pd.DataFrame:
    """
    Read a spreadsheet file into a pandas DataFrame, keeping only columns

    Args:
        filepath: str        The name of the file to read.
        sheet:    str        The name of the sheet to read (if excel).
        rows:     int        The number of rows to read (if known).
        columns:  List[str]  The columns to keep

    Returns:
        pd.DataFrame:       The data read from the file.
    """
    # read the spreadsheet
    df = read_spreadsheet(filepath, sheet=sheet, rows=rows)

    # eliminate all but requested columns
    if columns:
        for col in df.columns:
            if col not in columns:
                df.drop(columns=[col], inplace=True)

    return df


def construct_outpath(df_row: pd.Series,
                      root_dir: str,
                      data_dir: str,
                      project: str,
                      ext: str) -> str:
    """
    Construct full path to an input video file.

    Parameters:
        df_row: pd.Series   Row of the DataFrame containing the metadata (filename, etc.).
        root_dir: str       Root path to all versions of the data.
        data_dir: str           Data folder name, e.g., 'data'
        project: str        Project name, e.g., 'FAMLI2', etc.
        ext: str            Extension of the data file.

    Returns:
        str:                Full path to the data file.
    """
    # get the file basename
    file_basename = os.path.splitext(df_row['filename'])[0]
    outpath = os.path.join(
        root_dir,
        data_dir,
        project,
        df_row['exam_dir'],
        f"{file_basename}{ext}")
    # return the full path to the data file
    return outpath


def merge_dfs(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        merge_col: str,
        sort_col: str,
        columns: Union[None, int, List[str]] = None,
) -> pd.DataFrame:
    """
    Merge two dataframes by taking all columns from the first dataframe
    and appending the given columns from the second dataframe.

    Args:
        df1 (pd.DataFrame):            First dataframe
        df2 (pd.DataFrame):            Second dataframe
        merge_col (str):               Column name to merge on.
        sort_col (str):                Column name to sort by.
                                       First sort column is this one,
                                           second sort column is the merge_col.
        columns (list, int, optional): List of columns to append from df2.
                                           If None, use all columns from df2.
                                           If integer (e.g., N) use the last N columns.
                                           If list, use the specified columns.

    Returns:
        pd.DataFrame:                  Merged dataframe
    """
    # Sort dataframes by merge_col
    df1.sort_values(by=[merge_col], inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df2.sort_values(by=[merge_col], inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # Make sure merge_col columns are identical
    if not all(df1[merge_col].values == df2[merge_col].values):
        raise ValueError(f"Merge columns '{merge_col}' " +
                         f"do not match in both dataframes.")

    # Get the columns to merge from the second dataframe
    if columns is None:
        cols_df2 = df2.columns.tolist()
    elif isinstance(columns, int):
        cols_df2 = df2.columns[-columns:].tolist()
    elif isinstance(columns, list):
        cols_df2 = columns
    else:
        raise ValueError("Parameter 'columns' must be None, " +
                         "an integer, or a list of strings.")

    # Create a new dataframe with all columns from df1
    # merged_df = df1.copy()
    merged_df = pd.DataFrame(columns=list(df1.columns) + cols_df2)

    # Populate the df1 columns
    for col in list(df1.columns):
        merged_df[col] = df1[col].copy()

    # Add the request columns from df2
    for col in cols_df2:
        merged_df[col] = df2[col].copy()

    # Sort the merged dataframe by the specified sort_col and merge_col
    merged_df.sort_values(by=[sort_col, merge_col], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df
