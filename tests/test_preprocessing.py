import numpy as np
import pytest
from unittest.mock import patch
import sys
import os
# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.load_data import load_trial, load_metadata, get_dataset_split
from preprocessing.preprocessing import (
    get_bad_mask,
    preprocess_eeg_trial,
    interpolate_bad_samples,
    extract_clean_windows,
    preprocess_time_domain,
    preprocess_frequency_domain,
    preprocess_deep_learning,
    preprocess_eeg_with_mne_time_domain,
    preprocess_eeg_with_mne_freq_domain,
)

@pytest.fixture
def mock_trial():
    n_samples = 500
    data = {
        'Time': np.linspace(0, 2, n_samples),
        **{f'EEG{i+1}': np.random.randn(n_samples) for i in range(8)},
        **{f'Acc{i}': np.random.randn(n_samples) * 2 for i in 'XYZ'},
        **{f'Gyro{i}': np.random.randn(n_samples) * 2 for i in range(1, 4)},
        'Battery': np.full(n_samples, 100),
        'Validation': np.ones(n_samples)
    }
    return data

@patch("preprocessing.load_trial")
def test_bad_mask_detection(mock_load):
    df = {
        'AccX': [0, 30], 'AccY': [0, 0], 'AccZ': [0, 0],
        'Gyro1': [0, 0], 'Gyro2': [0, 10], 'Gyro3': [0, 0],
        'Battery': [100, 10],
        'Validation': [1, 0]
    }
    import pandas as pd
    bad_mask = get_bad_mask(pd.DataFrame(df))
    assert bad_mask.tolist() == [False, True]

@patch("preprocessing.load_trial")
def test_preprocess_eeg_trial_modes(mock_load, mock_trial):
    from pandas import DataFrame
    mock_load.return_value = DataFrame(mock_trial)

    dummy_row = {'subject_id': 1, 'trial': 1, 'task': 'A', 'trial_session': 'S1'}

    result_time = preprocess_eeg_trial(dummy_row, use_case='time_domain')
    assert isinstance(result_time, list)
    assert all(r.shape[0] == 8 for r in result_time)

    result_freq = preprocess_eeg_trial(dummy_row, use_case='frequency_domain')
    assert isinstance(result_freq, np.ndarray)
    assert result_freq.shape[0] == 8

    result_deep, mask = preprocess_eeg_trial(dummy_row, use_case='deep_learning')
    assert result_deep.shape == mask.shape[::-1]

def test_interpolation_behavior():
    eeg = np.random.randn(8, 100)
    mask = np.zeros(100, dtype=bool)
    mask[10:20] = True
    eeg[:, 10:20] = 0
    interpolated = interpolate_bad_samples(eeg, mask)
    assert not np.any(interpolated[:, 10:20] == 0)

def test_window_extraction():
    eeg = np.random.randn(8, 500)
    bad_mask = np.zeros(500, dtype=bool)
    bad_mask[100:150] = True
    segments = extract_clean_windows(eeg, bad_mask, window_size_sec=1, stride_sec=0.5, fs=250)
    assert all(seg.shape == (8, 250) for seg in segments)

def test_preprocess_time_domain_shape():
    eeg = np.random.randn(8, 500)
    eeg_proc, fs_out = preprocess_time_domain(eeg, fs=250, downsample_to=100)
    assert eeg_proc.shape[1] == 100 * (500 / 250)

def test_preprocess_frequency_domain_shape():
    eeg = np.random.randn(8, 500)
    eeg_proc, _ = preprocess_frequency_domain(eeg, fs=250, downsample_to=100)
    assert eeg_proc.shape[1] == 100 * (500 / 250)

def test_preprocess_deep_learning_properties():
    eeg = np.random.randn(8, 500)
    eeg_proc, _ = preprocess_deep_learning(eeg, fs=250)
    assert np.isclose(np.mean(eeg_proc), 0, atol=1)

@patch("preprocessing.load_trial")
@patch("preprocessing.get_dataset_split", return_value='train')
def test_mne_time_preprocessing_shape(mock_split, mock_load, mock_trial):
    from pandas import DataFrame
    mock_load.return_value = DataFrame(mock_trial)
    dummy_row = {'id': 1, 'subject_id': 1, 'trial': 1, 'task': 'A', 'trial_session': 'S1'}
    result, fs = preprocess_eeg_with_mne_time_domain(dummy_row, window_size_sec=2.0, stride_sec=1.0)
    assert result.ndim == 3  # (windows, channels, time)

@patch("preprocessing.load_trial")
@patch("preprocessing.get_dataset_split", return_value='train')
def test_mne_freq_preprocessing_shape(mock_split, mock_load, mock_trial):
    from pandas import DataFrame
    mock_load.return_value = DataFrame(mock_trial)
    dummy_row = {'id': 1, 'subject_id': 1, 'trial': 1, 'task': 'A', 'trial_session': 'S1'}
    result, freqs, fs = preprocess_eeg_with_mne_freq_domain(dummy_row, spec_fn='welch')
    assert result.ndim == 3  # (windows, channels, freqs)
    assert len(freqs) > 0
