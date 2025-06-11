import os
import pandas as pd
import logging

# Constants
SAMPLES_PER_TRIAL = {
    'MI': 2250,
    'SSVEP': 1750
}


# Setup logging
LOG_DIR = "../outputs/logs"
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "load_data.log"),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_metadata(base_path):
    """
    Load train, validation, and test metadata CSV files.

    Parameters
    ----------
    base_path : str
        Path to the directory containing train.csv, validation.csv, and test.csv.

    Returns
    -------
    dict of str: pd.DataFrame
        Dictionary with keys 'train', 'validation', and 'test' containing metadata.
    """
    logging.info(f"Loading metadata from {base_path}")
    try:
        return {
            'train': pd.read_csv(os.path.join(base_path, 'train.csv')),
            'validation': pd.read_csv(os.path.join(base_path, 'validation.csv')),
            'test': pd.read_csv(os.path.join(base_path, 'test.csv'))
        }
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        raise

def load_eeg_file(base_path, task, dataset, subject_id, session):
    """
    Load EEG data CSV for a given subject and session.

    Parameters
    ----------
    base_path : str
    task : str
        Either 'MI' or 'SSVEP'.
    dataset : str
        One of 'train', 'validation', or 'test'.
    subject_id : int
    session : int

    Returns
    -------
    pd.DataFrame
        EEG data.
    """
    path = os.path.join(base_path, task, dataset, f"S{subject_id}", str(session), 'EEGdata.csv')
    logging.info(f"Loading EEG data from {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Failed to load EEG file: {e}")
        raise

def load_trial(row, dataset, base_path):
    """
    Load EEG data for a single trial.

    Parameters
    ----------
    row : pd.Series
        A row from metadata containing task, subject_id, trial_session, and trial.
    dataset : str
        One of 'train', 'validation', or 'test'.
    base_path : str

    Returns
    -------
    pd.DataFrame
        EEG data for the trial.
    """
    try:
        eeg_data = load_eeg_file(base_path, row['task'], dataset, row['subject_id'], row['trial_session'])
        samples_per_trial = SAMPLES_PER_TRIAL[row['task']]
        trial_num = int(row['trial'])
        start = (trial_num - 1) * samples_per_trial
        end = start + samples_per_trial
        logging.info(f"Extracted trial {trial_num} from session {row['trial_session']} for subject {row['subject_id']}")
        return eeg_data.iloc[start:end]
    except Exception as e:
        logging.error(f"Failed to load trial: {e}")
        raise

def load_all_trials_for_task_split(base_path, metadata_df, task_name):
    """
    Load all trials for a specific task in a dataset split.

    Parameters
    ----------
    base_path : str
    metadata_df : pd.DataFrame
        DataFrame containing metadata with columns: task, subject_id, trial_session, trial.
    task_name : str
        Either 'MI' or 'SSVEP'.

    Returns
    -------
    list of pd.DataFrame
        List of EEG trial dataframes.
    """
    return [load_trial(row, get_dataset_split(row['id']), base_path) for _, row in metadata_df.iterrows() if row['task'] == task_name]

def load_all_mi_data(base_path, metadata):
    """
    Load all Motor Imagery (MI) data.

    Parameters
    ----------
    base_path : str
    metadata : pd.DataFrame

    Returns
    -------
    list of pd.DataFrame
    """
    return load_all_trials_for_task_split(base_path, metadata, 'MI')

def load_all_ssvep_data(base_path, metadata):
    """
    Load all Steady-State Visually Evoked Potential (SSVEP) data.

    Parameters
    ----------
    base_path : str
    metadata : pd.DataFrame

    Returns
    -------
    list of pd.DataFrame
    """
    return load_all_trials_for_task_split(base_path, metadata, 'SSVEP')

def load_all_data(base_path, metadata):
    """
    Load all EEG trial data (MI and SSVEP).

    Parameters
    ----------
    base_path : str
    metadata : pd.DataFrame

    Returns
    -------
    list of pd.DataFrame
    """
    return [load_trial(row, get_dataset_split(row['id']), base_path) for _, row in metadata.iterrows()]

def load_subject_trials(base_path, metadata, subject_id):
    """
    Load all EEG trials for a given subject.

    Parameters
    ----------
    base_path : str
    metadata : pd.DataFrame
    subject_id : int

    Returns
    -------
    list of list of pd.DataFrame
    """
    subject_rows = metadata[metadata['subject_id'] == subject_id]
    logging.info(f"Loading all trials for subject {subject_id}")
    return [[load_trial(row, get_dataset_split(row['id']), base_path) for _, row in subject_rows.iterrows()]]

def load_by_session_and_trial(base_path, metadata, subject_id, session, trial_num):
    """
    Load EEG data for a specific session and trial number for a subject.

    Parameters
    ----------
    base_path : str
    metadata : pd.DataFrame
    subject_id : int
    session : int
    trial_num : int

    Returns
    -------
    pd.DataFrame or None
    """
    rows = metadata[
        (metadata['subject_id'] == subject_id) &
        (metadata['trial_session'] == session) &
        (metadata['trial'] == trial_num)
    ]
    if not rows.empty:
        logging.info(f"Loading trial {trial_num} from session {session} for subject {subject_id}")
        return load_trial(rows.iloc[0], get_dataset_split(rows.iloc[0]['id']), base_path)
    logging.warning(f"No data found for subject {subject_id}, session {session}, trial {trial_num}")
    return None

def get_dataset_split(id):
    """
    Determine dataset split based on ID.

    Parameters
    ----------
    id : int

    Returns
    -------
    str
        One of 'train', 'validation', or 'test'
    """
    if id <= 4800:
        return 'train'
    elif id <= 4900:
        return 'validation'
    else:
        return 'test'