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

# scripts/preprocess_audio.py
import os
import librosa
import numpy as np

def load_audio_dataset(audio_dir, sr=16000, return_ids=False):
    """
    Loads raw audio waveforms from a directory and infers labels from filenames.
    
    Args:
        audio_dir (str): Path to directory containing audio files.
        sr (int): Sampling rate to load audio.
        return_ids (bool): Whether to return the list of file IDs.
        
    Returns:
        X (list of np.array): Raw waveforms.
        y (np.array): Labels (0=Control, 1=Dementia).
        ids (list): File IDs (optional).
    """
    X = []
    y = []
    ids = []

    # Scan all files in the directory
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)

                # Load raw waveform
                signal, _ = librosa.load(file_path, sr=sr)

                # Label inference: customize this based on your filenames
                fname_lower = file.lower()
                if "dementia" in fname_lower:
                    label = 1
                elif "control" in fname_lower:
                    label = 0
                else:
                    # Unknown label → skip
                    continue

                X.append(signal)
                y.append(label)
                ids.append(os.path.splitext(file)[0])  # filename without extension

    X = np.array(X, dtype=object)  # variable-length waveforms
    y = np.array(y, dtype=int)

    if return_ids:
        return X, y, ids
    else:
        return X, y
