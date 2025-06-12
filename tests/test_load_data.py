import pytest
import pandas as pd
from unittest import mock
from unittest.mock import patch, MagicMock
import sys
import os
# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils.load_data import load_trial, load_metadata, get_dataset_split


# Sample metadata mock
@pytest.fixture
def sample_metadata():
    return pd.DataFrame([
        {'id': 1, 'task': 'MI', 'subject_id': 1, 'trial_session': 1, 'trial': 1},
        {'id': 4801, 'task': 'SSVEP', 'subject_id': 2, 'trial_session': 1, 'trial': 1},
    ])


# Test get_dataset_split
@pytest.mark.parametrize("id_val, expected", [
    (100, 'train'),
    (4800, 'train'),
    (4801, 'validation'),
    (4900, 'validation'),
    (4901, 'test'),
])
def test_get_dataset_split(id_val, expected):
    assert load_data.get_dataset_split(id_val) == expected


# Test load_metadata with mocking
@patch("pandas.read_csv")
def test_load_metadata(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({'id': [1]})
    result = load_data.load_metadata("fake_path")
    assert set(result.keys()) == {"train", "validation", "test"}
    assert all(isinstance(df, pd.DataFrame) for df in result.values())


# Test load_eeg_file with mock
@patch("pandas.read_csv")
def test_load_eeg_file(mock_read_csv):
    dummy_df = pd.DataFrame({'channel1': range(100)})
    mock_read_csv.return_value = dummy_df
    result = load_data.load_eeg_file("base", "MI", "train", 1, 1)
    assert isinstance(result, pd.DataFrame)


# Test load_trial slices correctly
@patch("src.utils.load_data.load_eeg_file")
def test_load_trial(mock_load_eeg_file, sample_metadata):
    full_data = pd.DataFrame({'ch': range(3000)})
    mock_load_eeg_file.return_value = full_data
    row = sample_metadata.iloc[0]
    trial_data = load_data.load_trial(row, "train", "base")
    assert len(trial_data) == load_data.SAMPLES_PER_TRIAL[row['task']]


# Test load_by_session_and_trial returns expected data
@patch("src.utils.load_data.load_trial")
def test_load_by_session_and_trial(mock_load_trial, sample_metadata):
    mock_load_trial.return_value = pd.DataFrame({'ch': range(10)})
    result = load_data.load_by_session_and_trial("base", sample_metadata, 1, 1, 1)
    assert isinstance(result, pd.DataFrame)

    # Nonexistent session/trial
    result_none = load_data.load_by_session_and_trial("base", sample_metadata, 99, 1, 1)
    assert result_none is None


# Test load_all_trials_for_task_split
@patch("src.utils.load_data.load_trial")
def test_load_all_trials_for_task_split(mock_load_trial, sample_metadata):
    mock_load_trial.return_value = pd.DataFrame({'ch': range(10)})
    result = load_data.load_all_trials_for_task_split("base", sample_metadata, "MI")
    assert isinstance(result, list)
    assert all(isinstance(r, pd.DataFrame) for r in result)


# Test load_all_data
@patch("src.utils.load_data.load_trial")
def test_load_all_data(mock_load_trial, sample_metadata):
    mock_load_trial.return_value = pd.DataFrame({'ch': range(10)})
    result = load_data.load_all_data("base", sample_metadata)
    assert len(result) == len(sample_metadata)


# Test load_subject_trials
@patch("src.utils.load_data.load_trial")
def test_load_subject_trials(mock_load_trial, sample_metadata):
    mock_load_trial.return_value = pd.DataFrame({'ch': range(10)})
    result = load_data.load_subject_trials("base", sample_metadata, 1)
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert all(isinstance(df, pd.DataFrame) for df in result[0])
