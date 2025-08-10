import os
import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=13, max_pad_len=300):
    # Load the audio
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Convert to decibel scale
    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    # Pad/truncate to fixed length
    if mfcc_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc_db.shape[1]
        mfcc_db = np.pad(mfcc_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_db = mfcc_db[:, :max_pad_len]

    return mfcc_db

def load_audio_dataset(root_dir, n_mfcc=13, max_pad_len=300, return_ids=False):
    

    X = []
    y = []
    ids = []

    for label_name, label_id in [('control', 1), ('dementia', 0)]:
        class_dir = os.path.join(root_dir, label_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.mp3'):
                filepath = os.path.join(class_dir, filename)
                mfcc = extract_mfcc(filepath, n_mfcc, max_pad_len)
                X.append(mfcc)
                y.append(label_id)
                ids.append(filename.replace(".mp3", ""))

    X = np.array(X)
    y = np.array(y)

    return (X, y, ids) if return_ids else (X, y)
