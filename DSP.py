import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Define EEG bands
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

results = []

# Loop through each subject
for subj in range(1, 6):
    # Load training and testing data
    train = loadmat(f'/Users/refobic/Downloads/Project/Data/train_data_{subj}.mat')
    test = loadmat(f'/Users/refobic/Downloads/Project/Data/test_data_{subj}.mat')
    
    data_train = train['data']  # trials x samples x channels
    labels_train = train['labels'].squeeze()
    data_test = test['data']
    labels_test = test['labels'].squeeze()
    
    fs = float(train['fs'].squeeze())
    channels = [str(c[0]) for c in train['channels'].squeeze()]
    
    # Common Average Reference (CAR) filtering
    def apply_car(data):
        return data - np.mean(data, axis=2, keepdims=True)
    
    data_train_car = apply_car(data_train)
    data_test_car = apply_car(data_test)
    
    # Feature extraction: compute band power
    n_trials, n_samples, n_channels = data_train_car.shape
    n_bands = len(bands)
    features_train = np.zeros((n_trials, n_channels, n_bands))
    features_test = np.zeros((data_test_car.shape[0], n_channels, n_bands))
    
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    
    # Helper to extract band-power features from data
    def extract_features(data_car):
        trials = data_car.shape[0]
        feats = np.zeros((trials, n_channels, n_bands))
        for t in range(trials):
            for ch in range(n_channels):
                sig = data_car[t, :, ch]
                power_spectrum = np.abs(np.fft.rfft(sig))**2
                for b_idx, (bname, (low, high)) in enumerate(bands.items()):
                    idx = np.where((freqs >= low) & (freqs < high))
                    feats[t, ch, b_idx] = power_spectrum[idx].mean()
        return feats
    
    features_train = extract_features(data_train_car)
    features_test = extract_features(data_test_car)
    
    # Classification scenarios
    # 1. Single channel + single band
    for ch_idx, ch_name in enumerate(channels):
        for b_idx, b_name in enumerate(bands):
            X_tr = features_train[:, ch_idx, b_idx].reshape(-1, 1)
            X_te = features_test[:, ch_idx, b_idx].reshape(-1, 1)
            best_acc, best_k = 0, 1
            for K in range(1, 11):
                knn = KNeighborsClassifier(n_neighbors=K)
                knn.fit(X_tr, labels_train)
                acc = knn.score(X_te, labels_test)
                if acc > best_acc:
                    best_acc, best_k = acc, K
            results.append({
                'Subject': subj,
                'Scenario': 'Single Channel & Band',
                'Channel': ch_name,
                'Band': b_name,
                'Best K': best_k,
                'Accuracy': best_acc
            })
    
    # 2. Single channel, all bands
    for ch_idx, ch_name in enumerate(channels):
        X_tr = features_train[:, ch_idx, :]
        X_te = features_test[:, ch_idx, :]
        best_acc, best_k = 0, 1
        for K in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=K)
            knn.fit(X_tr, labels_train)
            acc = knn.score(X_te, labels_test)
            if acc > best_acc:
                best_acc, best_k = acc, K
        results.append({
            'Subject': subj,
            'Scenario': 'Single Channel, All Bands',
            'Channel': ch_name,
            'Band': 'All',
            'Best K': best_k,
            'Accuracy': best_acc
        })
    
    # 3. Single band, all channels
    for b_idx, b_name in enumerate(bands):
        X_tr = features_train[:, :, b_idx]
        X_te = features_test[:, :, b_idx]
        best_acc, best_k = 0, 1
        for K in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=K)
            knn.fit(X_tr, labels_train)
            acc = knn.score(X_te, labels_test)
            if acc > best_acc:
                best_acc, best_k = acc, K
        results.append({
            'Subject': subj,
            'Scenario': 'Single Band, All Channels',
            'Channel': 'All',
            'Band': b_name,
            'Best K': best_k,
            'Accuracy': best_acc
        })
    
    # 4. All features (all channels & all bands)
    X_tr = features_train.reshape(n_trials, n_channels * n_bands)
    X_te = features_test.reshape(data_test_car.shape[0], n_channels * n_bands)
    best_acc, best_k = 0, 1
    for K in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(X_tr, labels_train)
        acc = knn.score(X_te, labels_test)
        if acc > best_acc:
            best_acc, best_k = acc, K
    results.append({
        'Subject': subj,
        'Scenario': 'All Channels & All Bands',
        'Channel': 'All',
        'Band': 'All',
        'Best K': best_k,
        'Accuracy': best_acc
    })

# Display results as a DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv('/Users/refobic/Downloads/Project/EEG_KNN_Results.csv', index=False)

