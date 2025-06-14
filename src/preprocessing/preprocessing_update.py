import logging
import mne
import numpy as np

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