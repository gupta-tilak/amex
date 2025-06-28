import pandas as pd
import os

# DATA_DIR should point to the dataset directory where the parquet files are located
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

# for kaggle uncomment this
# DATA_DIR = os.path.dirname(os.path.dirname(__file__))

# File names (update if file names change)
TRAIN_FILE = 'train_data_converted.parquet'
TEST_FILE = 'test_data_converted.parquet'
OFFER_METADATA_FILE = 'offer_metadata_converted.parquet'
ADD_EVENT_FILE = 'add_event_converted.parquet'
ADD_TRANS_FILE = 'add_trans_converted.parquet'
DATA_DICTIONARY_FILE = 'data_dictionary.csv'


def load_train_data():
    """Load the training data."""
    return pd.read_parquet(os.path.join(DATA_DIR, TRAIN_FILE))

def load_test_data():
    """Load the test data."""
    return pd.read_parquet(os.path.join(DATA_DIR, TEST_FILE))

def load_offer_metadata():
    """Load the offer metadata."""
    return pd.read_parquet(os.path.join(DATA_DIR, OFFER_METADATA_FILE))

def load_add_event():
    """Load the add_event data."""
    return pd.read_parquet(os.path.join(DATA_DIR, ADD_EVENT_FILE))

def load_add_trans():
    """Load the add_trans data."""
    return pd.read_parquet(os.path.join(DATA_DIR, ADD_TRANS_FILE))

def load_data_dictionary():
    """Load the data dictionary (for reference)."""
    return pd.read_csv(os.path.join(DATA_DIR, DATA_DICTIONARY_FILE))


def load_all_data():
    """Load all datasets and return as a dictionary."""
    return {
        'train': load_train_data(),
        'test': load_test_data(),
        'offer_metadata': load_offer_metadata(),
        'add_event': load_add_event(),
        'add_trans': load_add_trans(),
        'data_dictionary': load_data_dictionary(),
    }

# Example usage (for testing):
# if __name__ == "__main__":
#     data = load_all_data()
#     for k, v in data.items():
#         print(f"{k}: {v.shape}")
