import pandas as pd
import numpy as np
import mne
import logging
import scipy.signal as signal
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.load_data import load_trial, load_metadata, get_dataset_split

# Configure module-level logger
logger = logging.getLogger(__name__)
# At startup of your application configure handlers/levels:
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_bad_mask(df, acc_thresh=15, gyro_thresh=5, battery_thresh=15):
    """
    Generate a boolean mask of "bad" time points in trial data.

    Bad samples are those where:
      - Acceleration magnitude > acc_thresh
      - Gyro magnitude > gyro_thresh
      - Battery voltage < battery_thresh
      - Validation flag == 0

    Returns:
        bad_mask (ndarray of bool): True for bad samples
    """
    logger.debug("Computing bad mask with thresholds: acc=%s, gyro=%s, battery=%s",
                 acc_thresh, gyro_thresh, battery_thresh)
    acc = df[['AccX', 'AccY', 'AccZ']].values
    gyro = df[['Gyro1', 'Gyro2', 'Gyro3']].values
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    mask = (
        (acc_mag > acc_thresh) |
        (gyro_mag > gyro_thresh) |
        (df['Battery'].values < battery_thresh) |
        (df['Validation'].values == 0)
    )
    logger.info("Bad mask created: %d bad samples out of %d total",
                mask.sum(), len(mask))
    return mask


def plot_trial_with_annotations(row, fs=250, max_channels=8):
    """
    Plots EEG channels + accel norm + gyro norm + validation.
    Highlights bad samples using vertical lines.
    
    Parameters:
        eeg_dict: dict from load_trial() — contains 'eeg', 'acc', 'gyro', 'validation'
        bad_mask: boolean array of shape (time,) where True = bad sample
        fs: sampling frequency (default: 250Hz)
        max_channels: number of EEG channels to plot (for clarity)
    """
    eeg_data = load_trial(row, 'train', base_path)
    bad_mask = get_bad_mask(eeg_data)
    
    eeg = eeg_data.iloc[:, 1:9].values.T
    acc = eeg_data.iloc[:, 9:12].values.T
    gyro = eeg_data.iloc[:, 12:15].values.T
    val = eeg_data.iloc[:, -1].values

    time = np.arange(eeg.shape[1]) / fs
    acc_norm = np.linalg.norm(acc, axis=0)
    gyro_norm = np.linalg.norm(gyro, axis=0)

    n_channels = min(max_channels, eeg.shape[0])
    fig, axs = plt.subplots(n_channels + 3, 1, figsize=(15, 2 * (n_channels + 3)), sharex=True)

    # EEG channels
    for ch in range(n_channels):
        axs[ch].plot(time, eeg[ch], label=f'EEG ch{ch+1}')
        axs[ch].set_ylabel(f'Ch {ch+1}')
        axs[ch].legend(loc='upper right')
        axs[ch].grid(True)

    # Acc norm
    axs[n_channels].plot(time, acc_norm, color='orange', label='Accel norm')
    axs[n_channels].set_ylabel('Accel')
    axs[n_channels].legend(loc='upper right')
    axs[n_channels].grid(True)

    # Gyro norm
    axs[n_channels + 1].plot(time, gyro_norm, color='purple', label='Gyro norm')
    axs[n_channels + 1].set_ylabel('Gyro')
    axs[n_channels + 1].legend(loc='upper right')
    axs[n_channels + 1].grid(True)

    # Validation
    axs[n_channels + 2].plot(time, val, color='green', label='Validation flag')
    axs[n_channels + 2].set_ylabel('Valid')
    axs[n_channels + 2].legend(loc='upper right')
    axs[n_channels + 2].grid(True)

    # Bad sample markers
    bad_indices = np.where(bad_mask)[0]

    # Group contiguous bad indices into regions
    bad_groups = []
    for k, g in groupby(enumerate(bad_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        bad_groups.append((group[0], group[-1]))

    # Highlight regions
    for ax in axs:
        for start, end in bad_groups:
            ax.axvspan(start/fs, end/fs, color='red', alpha=0.2)

        axs[-1].set_xlabel('Time (s)')
    plt.suptitle(f"EEG Trial: {row['subject_id']} | Task: {row['task']} | Session: {row['trial_session']} | Trial: {row['trial']}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
def preprocess_eeg_trial(row, use_case='time_domain', fs=250,
                         interpolation_kind='linear',
                         window_size_sec=2, stride_sec=0.5):
    """
    Preprocess EEG data for modeling or feature extraction.
    
    Parameters:
        eeg_dict (dict): Dictionary with keys 'eeg', 'acc', 'gyro', etc.
        bad_mask (ndarray): Boolean mask of shape (time,) indicating bad timepoints.
        use_case (str): One of 'time_domain', 'frequency_domain', or 'deep_learning'.
        fs (int): Sampling frequency in Hz.
        interpolation_kind (str): 'linear', 'nearest', etc.
        window_size_sec (float): Size of clean window (in seconds) for 'time_domain'.
        stride_sec (float): Stride (in seconds) between windows.
        
    Returns:
        Depending on use_case:
            - List of clean windows (time/frequency)
            - Preprocessed raw EEG for deep learning
            - Optional mask for deep learning
    """
    eeg_data = load_trial(row, 'train', base_path)
    bad_mask = get_bad_mask(eeg_data)
    eeg = eeg_data.iloc[:, 1:9].values.T  # shape (channels, time)

    if use_case == 'time_domain':
        # Remove windows with any bad samples
        return extract_clean_windows(eeg, bad_mask, window_size_sec, stride_sec, fs)

    elif use_case == 'frequency_domain':
        # Interpolate bad samples to maintain continuity
        eeg_clean = interpolate_bad_samples(eeg, bad_mask, kind=interpolation_kind)
        return eeg_clean

    elif use_case == 'deep_learning':
        # Interpolate or zero out and provide a mask
        eeg_interp = interpolate_bad_samples(eeg, bad_mask, kind=interpolation_kind)
        mask = (~bad_mask).astype(float)  # 1 for good, 0 for bad
        return eeg_interp, mask  # mask shape (time,)

    else:
        raise ValueError(f"Unknown use_case: {use_case}")
    

def interpolate_bad_samples(eeg, bad_mask, kind='linear'):
    """
    Interpolate across bad time points in multi-channel EEG using 1D interpolation.

    Args:
        eeg (ndarray): shape (n_channels, n_times)
        bad_mask (ndarray): boolean mask of length n_times
        kind (str): interpolation method e.g. 'linear', 'nearest'

    Returns:
        eeg_clean (ndarray): same shape, with bad points interpolated
    """
    logger.debug("Interpolating bad samples: kind=%s", kind)
    eeg_clean = eeg.copy()
    time = np.arange(eeg.shape[1])
    for ch in range(eeg.shape[0]):
        good = ~bad_mask
        if good.sum() < 2:
            eeg_clean[ch, bad_mask] = 0
            logger.warning("Channel %d has <2 good points, zeroed bad samples", ch)
            continue
        interp_fn = interpolate.interp1d(
            time[good], eeg[ch, good], kind=kind,
            fill_value='extrapolate', bounds_error=False
        )
        eeg_clean[ch, bad_mask] = interp_fn(time[bad_mask])
    logger.info("Interpolation complete")
    return eeg_clean

def extract_clean_windows(eeg, bad_mask, window_size_sec, stride_sec, fs):
    """
    Extract time-domain segments with no bad samples.

    Args:
        eeg (ndarray): shape (n_ch, n_times)
        bad_mask (ndarray): same length as second dim
        window_size_sec (float)
        stride_sec (float)
        fs (int): sampling frequency

    Returns:
        segments (list of ndarray): each shape (n_ch, window_samples)
    """
    win = int(window_size_sec * fs)
    stride = int(stride_sec * fs)
    logger.info("Extracting windows: size %ds (%d), stride %ds (%d)", window_size_sec, win, stride_sec, stride)
    segments = []
    for st in range(0, eeg.shape[1] - win + 1, stride):
        segment_mask = bad_mask[st:st + win]
        if not segment_mask.any():
            segments.append(eeg[:, st:st + win])
    logger.info("Extracted %d clean windows", len(segments))
    return segments

def plot_eeg_channels(eeg_data, fs=250, channel_names=None, figsize=(12, 10)):
    """
    Plot EEG channels on separate axes.

    Parameters:
        eeg_data (ndarray): EEG array of shape (num_channels, samples)
        fs (int): Sampling frequency in Hz
        channel_names (list): Optional list of channel names
        figsize (tuple): Figure size for the plot
    """
    num_channels, num_samples = eeg_data.shape
    time = np.arange(num_samples) / fs

    if channel_names is None:
        channel_names = [f"Ch {i+1}" for i in range(num_channels)]

    fig, axs = plt.subplots(num_channels, 1, figsize=figsize, sharex=True)

    for i in range(num_channels):
        axs[i].plot(time, eeg_data[i])
        axs[i].set_ylabel(channel_names[i])
        axs[i].grid(True, linestyle='--', alpha=0.3)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("EEG Channels", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def bandpass_filter(data, lowcut, highcut, fs, order=4, min_length=30):
    """
    Apply zero-phase Butterworth bandpass filter to EEG data.
    If trial is too short, it will be skipped or zero-padded.

    Args:
        data (pd.DataFrame or np.ndarray): EEG signal (n_times, n_channels)
        lowcut (float): Low cutoff frequency (Hz)
        highcut (float): High cutoff frequency (Hz)
        fs (int): Sampling rate (Hz)
        order (int): Filter order
        min_length (int): Minimum number of timepoints required for filtering

    Returns:
        np.ndarray: Filtered EEG signal (n_times, n_channels) or None if too short
    """
    from scipy.signal import butter, filtfilt
    import numpy as np

    logger.debug("Bandpass filtering: %.2f–%.2f Hz, order %d", lowcut, highcut, order)

    # Drop irrelevant columns if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        non_eeg_columns = ['Time', 'AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3', 'Battery', 'Counter', 'Validation']
        eeg_data = data.drop(columns=[col for col in non_eeg_columns if col in data.columns], errors='ignore').to_numpy()
    else:
        eeg_data = data

    eeg_data = eeg_data.T  # (n_ch, n_times)
    n_ch, n_times = eeg_data.shape

    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    padlen = 3 * max(len(a), len(b))

    # Handle too-short signals
    if n_times <= padlen:
        logger.warning(f"Skipping trial: too short for filtering ({n_times} < {padlen})")
        return None  # or return np.zeros_like(eeg_data.T) to pad instead

    filtered = filtfilt(b, a, eeg_data, axis=1)
    logger.info("Filtering complete")
    return filtered.T  # back to (n_times, n_ch)



def remove_dc_offset(data):
    """
    Subtract channel-wise mean (DC offset).
    Converts to NumPy array if input is a DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        # Keep only numeric EEG channels and convert to NumPy
        data = data.select_dtypes(include=[np.number]).to_numpy()
    return data - np.mean(data, axis=0, keepdims=True)


def standardize(data):
    """
    Channel-wise standardization to zero mean and unit variance.

    Args:
        data (ndarray): shape (n_ch, n_times)
    Returns:
        standardized (ndarray)
    """
    logger.debug("Standardizing data")
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True) + 1e-8
    return (data - mean) / std

def clip_extremes(data, threshold=5):
    """
    Clip signal values to a symmetric absolute threshold.

    Args:
        data (ndarray)
        threshold (float)
    """
    logger.debug("Clipping extremes to ±%s", threshold)
    return np.clip(data, -threshold, threshold)

def preprocess_time_domain(eeg_data, fs=250, lowcut=1, highcut=45,
                           downsample_to=None, clip_thresh=None):
    """
    Preprocess raw EEG for time-domain use:
      - Remove DC
      - Bandpass filter
      - Optional clipping
      - Optional downsampling

    Args:
        eeg_data (ndarray): shape (n_ch, n_times)
        fs (int)
        lowcut (float), highcut (float)
        downsample_to (int or None)
        clip_thresh (float or None)

    Returns:
        eeg_processed (ndarray), fs_out (int)
    """
    logger.info("Time-domain preprocess start: %d channels, %d samples", *eeg_data.shape)
    eeg = remove_dc_offset(eeg_data)
    eeg = bandpass_filter(eeg, lowcut, highcut, fs)
    if clip_thresh is not None:
        eeg = clip_extremes(eeg, threshold=clip_thresh)
    if downsample_to and downsample_to < fs:
        eeg = signal.resample(eeg, int(eeg.shape[1] * downsample_to / fs), axis=1)
        fs = downsample_to
        logger.info("Downsampled to %d Hz", fs)
    logger.info("Time-domain preprocessing completed")
    return eeg, fs

def preprocess_frequency_domain(eeg_data, fs=250, lowcut=1, highcut=45,
                                downsample_to=None, clip_thresh=None):
    """
    Preprocess for frequency-domain analysis (e.g., bandpower):
      - DC removal, bandpass, optional clipping
      - Optional downsampling

    Returns:
        eeg_processed (ndarray), fs_out (int)
    """
    logger.info("Frequency-domain preprocess start")
    eeg = remove_dc_offset(eeg_data)
    eeg = bandpass_filter(eeg, lowcut, highcut, fs)
    if clip_thresh is not None:
        eeg = clip_extremes(eeg, threshold=clip_thresh)
    if downsample_to and downsample_to < fs:
        eeg = signal.resample(eeg, int(eeg.shape[1] * downsample_to / fs), axis=1)
        fs = downsample_to
        logger.info("Downsampled to %d Hz", fs)
    logger.info("Frequency-domain preprocessing done")
    return eeg, fs

def preprocess_deep_learning(eeg_data, fs=250, lowcut=1, highcut=45, clip_thresh=None):
    """
    Preprocess EEG for deep learning workflows:
      - DC removal, bandpass, optional clipping
      - Standardization (required for ML convergence)

    Args:
        eeg_data (ndarray), fs, lowcut, highcut, clip_thresh

    Returns:
        eeg_processed (ndarray), fs
    """
    logger.info("Deep-learning preprocess start")
    eeg = remove_dc_offset(eeg_data)
    eeg = bandpass_filter(eeg, lowcut, highcut, fs)
    if clip_thresh is not None:
        eeg = clip_extremes(eeg, threshold=clip_thresh)
    eeg = standardize(eeg)
    logger.info("Deep-learning preprocessing done")
    return eeg, fs

def preprocess_eeg_with_mne_time_domain(
    row,
    fs=250,
    window_size_sec=2.0,
    stride_sec=0.5,
    ch_names=None,
    l_freq=1.,
    h_freq=45.,
    notch_freq=50.,
    resample_to=None,
    max_bad_ratio=0.2,
    zscore=True,
    high_amp_thresh=5.0
):
    """
    Preprocess EEG trial data using MNE for time-domain feature extraction.

    Steps:
        1. Load raw EEG for given trial (handles interpolation of bad samples).
        2. Create MNE RawArray and set average reference.
        3. Apply notch (e.g., power line removal) and bandpass filtering.
        4. Resample the data if specified.
        5. Remove high-amplitude artifacts by zeroing out extreme deviations.
        6. Optionally z-score normalize each channel.
        7. Segment into overlapping windows, filtering out windows with too many bad samples.

    Parameters
    ----------
    row : pd.Series
        Metadata for a single trial (subject, session, trial, etc.)
    fs : int
        Original sampling frequency of the trial.
    window_size_sec : float
        Length of each segment in seconds.
    stride_sec : float
        Time step between consecutive segments in seconds.
    ch_names : list[str] or None
        Custom channel names. If None, defaults to ["Ch1", "Ch2", ..., "ChN"].
    l_freq : float
        Low cutoff frequency for bandpass filtering.
    h_freq : float
        High cutoff frequency for bandpass filtering.
    notch_freq : float or list
        Frequency (or list) to notch filter (e.g., power-line noise at 50 or 60 Hz).
    resample_to : int or None
        If specified, resample data to this sampling rate.
    max_bad_ratio : float
        Maximum allowed fraction of bad samples per window (0–1).
    zscore : bool
        If True, normalize channels to zero mean and unit variance.
    high_amp_thresh : float
        Artifact threshold in standard deviations—values beyond this are zeroed.

    Returns
    -------
    windows : np.ndarray
        3D array of shape (n_windows, n_channels, window_samples).
    final_fs : int
        Sampling rate after resampling.
    """
    logger.info("Starting MNE time-domain preprocessing for trial id=%s", row['id'])

    dataset = get_dataset_split(row['id'])
    eeg_data = load_trial(row, dataset, base_path)
    bad_mask = get_bad_mask(eeg_data)
    eeg = eeg_data.iloc[:, 1:9].values.T  # shape (channels, time)
    logger.debug("Loaded EEG data shape: %s", eeg.shape)

    # Interpolate bad samples
    eeg_cleaned = eeg.copy()
    for ch in range(eeg.shape[0]):
        good_idx = ~bad_mask
        if good_idx.sum() < 2:
            eeg_cleaned[ch] = 0.0
            logger.warning("Channel %d has insufficient good data—filled with zeros", ch)
        else:
            interp = interpolate.interp1d(
                np.flatnonzero(good_idx),
                eeg[ch, good_idx],
                kind='linear',
                fill_value='extrapolate'
            )
            eeg_cleaned[ch] = interp(np.arange(eeg.shape[1]))
    logger.info("Interpolation of bad samples completed")

    if ch_names is None:
        ch_names = [f"Ch{i+1}" for i in range(eeg.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(eeg_cleaned, info)

    raw.set_eeg_reference('average', projection=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    logger.debug("Notch and bandpass (%s–%s Hz) filters applied", notch_freq, (l_freq, h_freq))

    if resample_to:
        raw.resample(resample_to, verbose=False)
        logger.info("Resampled data from %s Hz to %s Hz", fs, resample_to)

    final_fs = int(raw.info['sfreq'])
    eeg_proc = raw.get_data()
    logger.debug("Raw data array shape after filtering: %s", eeg_proc.shape)

    # Remove high-amplitude artifacts
    stds = np.std(eeg_proc, axis=1, keepdims=True)
    artifact_mask = np.abs(eeg_proc) > (high_amp_thresh * stds)
    eeg_proc[artifact_mask] = 0.0
    logger.info("Zeroed out high-amplitude artifacts beyond %s stddev", high_amp_thresh)

    # Z-score normalization
    if zscore:
        means = np.mean(eeg_proc, axis=1, keepdims=True)
        stds = np.std(eeg_proc, axis=1, keepdims=True) + 1e-8
        eeg_proc = (eeg_proc - means) / stds
        logger.info("Z-score normalization applied per channel")

    if resample_to:
        # Downsample bad_mask to match new sampling rate
        resampled_mask = signal.resample(bad_mask.astype(float), eeg_proc.shape[1])
        bad_mask = resampled_mask > 0.5
        logger.debug("Resampled bad mask to match data length: %s", bad_mask.shape)

    win_len = int(window_size_sec * final_fs)
    stride_len = int(stride_sec * final_fs)
    windows = []
    logger.info("Segmenting data into windows of %s samples (stride %s)", win_len, stride_len)

    for start in range(0, eeg_proc.shape[1] - win_len + 1, stride_len):
        end = start + win_len
        window_bad_ratio = bad_mask[start:end].mean()
        if window_bad_ratio <= max_bad_ratio:
            windows.append(eeg_proc[:, start:end])
        else:
            logger.debug(
                "Dropped window %d–%d due to high bad ratio %.2f",
                start, end, window_bad_ratio
            )

    windows = np.stack(windows)
    logger.info("Generated %d windows (each %d samples)", windows.shape[0], win_len)

    return windows, final_fs

def preprocess_eeg_with_mne_freq_domain(
    row,
    fs=250,
    window_size_sec=2.0,
    stride_sec=0.5,
    ch_names=None,
    l_freq=1.,
    h_freq=45.,
    notch_freq=50.,
    resample_to=None,
    max_bad_ratio=0.2,
    zscore=True,
    high_amp_thresh=5.0
):
    """
    Preprocess EEG trial data using MNE for frequency-domain feature extraction.

    Steps:
        1. Load raw EEG for the trial, handle bad samples with interpolation.
        2. Create MNE RawArray and apply average referencing.
        3. Apply notch and bandpass filtering.
        4. Optional resampling.
        5. Remove high-amplitude outliers.
        6. Optional z-scoring of each channel.
        7. Segment the data into overlapping windows.
        8. Apply FFT to each window and keep power spectrum.

    Parameters
    ----------
    row : pd.Series
        Metadata row for a trial (contains 'id', etc.).
    fs : int
        Original sampling rate of the data.
    window_size_sec : float
        Size of each window in seconds.
    stride_sec : float
        Step size between windows in seconds.
    ch_names : list[str] or None
        List of channel names; defaults to generic names if None.
    l_freq : float
        Low cutoff frequency for bandpass filter.
    h_freq : float
        High cutoff frequency for bandpass filter.
    notch_freq : float or list
        Frequency/frequencies for notch filter (e.g., 50 or 60 Hz).
    resample_to : int or None
        Resample to this rate if specified.
    max_bad_ratio : float
        Max allowable proportion of bad samples in a window.
    zscore : bool
        Whether to z-score each channel before FFT.
    high_amp_thresh : float
        Threshold for zeroing out samples with large amplitude (in stddev).

    Returns
    -------
    freq_windows : np.ndarray
        Array of shape (n_windows, n_channels, freq_bins), power spectral features.
    freqs : np.ndarray
        Frequency values corresponding to the FFT bins.
    """
    logger.info("Starting MNE frequency-domain preprocessing for trial id=%s", row['id'])

    dataset = get_dataset_split(row['id'])
    eeg_data = load_trial(row, dataset, base_path)
    bad_mask = get_bad_mask(eeg_data)
    eeg = eeg_data.iloc[:, 1:9].values.T
    logger.debug("Loaded EEG data shape: %s", eeg.shape)

    # Interpolation of bad samples
    eeg_cleaned = eeg.copy()
    for ch in range(eeg.shape[0]):
        good_idx = ~bad_mask
        if good_idx.sum() < 2:
            eeg_cleaned[ch] = 0.0
            logger.warning("Channel %d has too few good samples; zero-filled", ch)
        else:
            interp = interpolate.interp1d(
                np.flatnonzero(good_idx),
                eeg[ch, good_idx],
                kind='linear',
                fill_value='extrapolate'
            )
            eeg_cleaned[ch] = interp(np.arange(eeg.shape[1]))
    logger.info("Interpolated bad samples")

    if ch_names is None:
        ch_names = [f"Ch{i+1}" for i in range(eeg.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(eeg_cleaned, info)

    raw.set_eeg_reference('average', projection=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    logger.debug("Filtering complete (notch and bandpass)")

    if resample_to:
        raw.resample(resample_to, verbose=False)
        logger.info("Resampled to %d Hz", resample_to)

    final_fs = int(raw.info['sfreq'])
    eeg_proc = raw.get_data()

    # Remove high-amplitude artifacts
    stds = np.std(eeg_proc, axis=1, keepdims=True)
    artifact_mask = np.abs(eeg_proc) > (high_amp_thresh * stds)
    eeg_proc[artifact_mask] = 0.0
    logger.info("High-amplitude artifacts removed (>%.1f std)", high_amp_thresh)

    # Z-score normalization
    if zscore:
        means = np.mean(eeg_proc, axis=1, keepdims=True)
        stds = np.std(eeg_proc, axis=1, keepdims=True) + 1e-8
        eeg_proc = (eeg_proc - means) / stds
        logger.info("Z-score normalization applied")

    if resample_to:
        resampled_mask = signal.resample(bad_mask.astype(float), eeg_proc.shape[1])
        bad_mask = resampled_mask > 0.5
        logger.debug("Bad mask resampled to match EEG length: %s", bad_mask.shape)

    win_len = int(window_size_sec * final_fs)
    stride_len = int(stride_sec * final_fs)
    freq_windows = []
    logger.info("Starting segmentation for FFT (%d sample windows)", win_len)

    for start in range(0, eeg_proc.shape[1] - win_len + 1, stride_len):
        end = start + win_len
        window_bad_ratio = bad_mask[start:end].mean()
        if window_bad_ratio <= max_bad_ratio:
            window = eeg_proc[:, start:end]
            # Compute power spectrum using FFT
            fft_data = np.fft.rfft(window, axis=1)
            power = np.abs(fft_data) ** 2
            freq_windows.append(power)
        else:
            logger.debug("Skipped FFT window %d–%d due to bad ratio %.2f", start, end, window_bad_ratio)

    freq_windows = np.stack(freq_windows)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / final_fs)
    logger.info("Generated %d FFT windows; freq resolution: %.2f Hz", freq_windows.shape[0], freqs[1] - freqs[0])

    return freq_windows, freqs

logging.basicConfig(level=logging.INFO)

def preprocess_trial(trial_data, accel_thresh=1.1, gyro_thresh=7.0, sfreq_eeg=250, battery_threshold=10):
    """
    Converts a single trial dataframe to an MNE Raw object with annotations for noisy segments.

    Parameters
    ----------
    trial_data : pd.DataFrame
        The trial dataframe containing EEG and auxiliary sensor data.
    accel_thresh : float, optional
        Threshold for acceleration magnitude (in g) above which segments are marked as bad, by default 1.1.
    gyro_thresh : float, optional
        Threshold for gyroscope magnitude (in deg/s) above which segments are marked as bad, by default 7.0.
    sfreq_eeg : int, optional
        Sampling frequency of the EEG data, by default 250 Hz.
    battery_threshold : float, optional
        Minimum acceptable battery level, by default 10.

    Returns
    -------
    raw : mne.io.Raw
        The MNE Raw object containing EEG and auxiliary channels.
    annotations : mne.Annotations | None
        Annotations marking bad segments, or None if no bad segments were found.
    """

    logging.info("Setting up EEG and AUX channel info...")
    eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    aux_channels = ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3', 'Validation', 'Battery']
    all_ch_names = eeg_channels + aux_channels
    ch_types = ['eeg'] * len(eeg_channels) + ['misc'] * len(aux_channels)

    info = mne.create_info(ch_names=all_ch_names, sfreq=sfreq_eeg, ch_types=ch_types)

    logging.info("Converting EEG data from µV to V...")
    trial_data_scaled = trial_data.copy()
    trial_data_scaled[eeg_channels] *= 1e-6  # µV → V

    raw_data = trial_data_scaled[all_ch_names].values.T
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    logging.info("Renaming and applying 10-20 montage...")
    raw.rename_channels({'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz'})
    raw.set_montage('standard_1020')

    onsets, durations, descriptions = [], [], []

    # --- Validation Check ---
    logging.info("Annotating bad signal quality (Validation == 0)...")
    bad_val = np.where(trial_data['Validation'] == 0)[0]
    _annotate_indices(bad_val, "bad_signal_quality", onsets, durations, descriptions, raw, sfreq_eeg)

    # --- Battery Check ---
    logging.info("Annotating low battery (< %d)...", battery_threshold)
    low_batt = np.where(trial_data['Battery'] < battery_threshold)[0]
    _annotate_indices(low_batt, "bad_battery_level", onsets, durations, descriptions, raw, sfreq_eeg)

    # --- Accelerometer Motion ---
    logging.info("Annotating high acceleration (>%0.2f g)...", accel_thresh)
    acc_norm = np.linalg.norm(trial_data[['AccX', 'AccY', 'AccZ']].values, axis=1)
    acc_bad = np.where(acc_norm > accel_thresh)[0]
    _annotate_indices(acc_bad, "bad_acceleration_motion", onsets, durations, descriptions, raw, sfreq_eeg)

    # --- Gyroscope Motion ---
    logging.info("Annotating high gyroscope motion (>%0.2f deg/s)...", gyro_thresh)
    gyro_norm = np.linalg.norm(trial_data[['Gyro1', 'Gyro2', 'Gyro3']].values, axis=1)
    gyro_bad = np.where(gyro_norm > gyro_thresh)[0]
    _annotate_indices(gyro_bad, "bad_gyro_motion", onsets, durations, descriptions, raw, sfreq_eeg)

    if onsets:
        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions,
                                      orig_time=raw.info['meas_date'])
        raw.set_annotations(annotations)
        logging.info("Total bad annotations: %d", len(annotations))
        return raw, annotations

    logging.info("No bad segments detected.")
    return raw, None


def _annotate_indices(indices, label, onsets, durations, descriptions, raw, sfreq):
    """
    Helper function to extract onset and duration segments from a list of indices
    and append them to the annotations lists.
    """
    if indices.size == 0:
        return
    diff = np.diff(indices)
    breaks = np.where(diff != 1)[0] + 1
    segments = np.split(indices, breaks)
    for seg in segments:
        if len(seg) == 0:
            continue
        onsets.append(float(raw.times[seg[0]]))
        durations.append(len(seg) / sfreq)
        descriptions.append(label)

def preprocess_and_epoch_trial(
    trial_data,
    sfreq_eeg=250,
    resample_sfreq=90,
    l_freq=1,
    h_freq=40,
    notch_freq=50,
    epoch_length=1.0,
    epoch_overlap=0.0,
    tmin=1.5,
    tmax=8,
    reject_bad_epochs=True,
    reference='average'
):
    """
    Preprocesses a single EEG trial and segments it into clean epochs.

    Parameters
    ----------
    trial_data : pd.DataFrame
        The raw trial dataframe containing EEG and auxiliary channels.
    sfreq_eeg : float, optional
        Original EEG sampling frequency in Hz, by default 250.
    resample_sfreq : float, optional
        Target sampling frequency in Hz, by default 90.
    l_freq : float, optional
        Low cut-off frequency for bandpass filter, by default 1 Hz.
    h_freq : float, optional
        High cut-off frequency for bandpass filter, by default 40 Hz.
    notch_freq : float, optional
        Frequency to apply notch filter at (e.g., 50Hz), by default 50.
    epoch_length : float, optional
        Duration of each epoch in seconds, by default 1.0.
    epoch_overlap : float, optional
        Overlap between epochs in seconds, by default 0.0.
    min_epoch_duration : float, optional
        Minimum clean segment duration to keep an epoch, by default 0.5s.
    reject_bad_epochs : bool, optional
        Whether to reject epochs that overlap with bad annotations, by default True.
    reference : str, optional
        EEG referencing method ('average', 'REST', etc.), by default 'average'.

    Returns
    -------
    epochs : mne.Epochs | None
        The clean segmented epochs or None if no valid segments found.
    raw : mne.io.Raw
        The final preprocessed raw object.
    """

    logging.info("Creating MNE Raw object...")
    raw, annotations = preprocess_trial(trial_data, sfreq_eeg=sfreq_eeg)
    
    logging.info("Setting EEG reference...")
    raw.set_eeg_reference(reference, verbose=False)

    logging.info("Applying notch filter at %.1f Hz...", notch_freq)
    raw.notch_filter(freqs=notch_freq, picks='eeg', fir_design='firwin', verbose=False)

    logging.info("Applying bandpass filter (%.1f–%.1f Hz)...", l_freq, h_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg', verbose=False)
    
    logging.info(f"Cropping to {tmin}-{tmax} seconds...")
    raw.crop(tmin=tmin, tmax=tmax)

    logging.info("Resampling to %.1f Hz...", resample_sfreq)
    raw_resampled = raw.copy().resample(resample_sfreq, npad='auto', method='fft')

    logging.info("Creating epochs (%.1fs, overlap %.1fs)...", epoch_length, epoch_overlap)
    events = mne.make_fixed_length_events(raw_resampled, start=0, duration=epoch_length - epoch_overlap)
    epochs = mne.Epochs(
        raw_resampled,
        events=events,
        event_id=1,
        tmin=0,
        tmax=epoch_length,
        baseline=None,
        detrend=0,
        reject_by_annotation=reject_bad_epochs,
        verbose=False,
        preload=True
    )

    if len(epochs) == 0:
        logging.warning("No valid epochs after rejection.")
        return None, raw_resampled

    logging.info("Finished processing trial: %d valid epochs", len(epochs))
    return epochs, raw_resampled