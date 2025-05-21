
# Ismaiel Sabet 900221277
# Mohamed El-Refai 900222334
# DSP project


import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from numpy.fft import rfft, rfftfreq

#Algorithm

    # Step 1: Compute the CAR filter to the data
    # Step 2: 
        #   2.1: For each electrode of the *7* electrodes,  
        #   2.2: For each trial of each class of attention (focused versus drowsy), compute the Fourier Transform and compute the power in the delta, theta, alpha, beta, and gamma
        #   2.3: Apply KNN
        #   2.4: Compute the classification error for each value of K (from 2.3)

bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

# For consistent ordering of bands in feature arrays
band_names_ordered = list(bands.keys())

def apply_car(data): 
    return data - np.mean(data, axis=2, keepdims=True) #because data has dimension of 3 and the count starts from 0 

def compute_band_power(signal, fs, band_definitions):
    # calculates power in specified frequency bands for a single signal trace
    # signal: 1D numpy array (time-series data for one trial, one channel)
    # fs: sampling frequency
    # band_definitions: dictionary defining band names and their (low_freq, high_freq)

    n = len(signal)
    if n == 0: # Handle empty signal case
        return {band_name: 0 for band_name in band_definitions}

    yf = rfft(signal)  # Compute the FFT
    power_spectrum = np.abs(yf)**2  # Compute power (magnitude squared)
    xf = rfftfreq(n, 1/fs)  # Get the frequencies

    # Calculate the power in each band
    band_powers = {}
    for band_name, (low_freq, high_freq) in band_definitions.items():
        band_mask = (xf >= low_freq) & (xf < high_freq) # Frequencies within the band

        # Calculate mean power in the band (if there are frequencies in the band)
        if np.any(band_mask) and len(power_spectrum[band_mask]) > 0: 
            band_powers[band_name] = np.mean(power_spectrum[band_mask]) 
        else:
            band_powers[band_name] = 0

    return band_powers

def extract_features(data_car, fs, band_definitions, ordered_band_names_list):
    # Extracts band power features for all trials and channels and bands
    # data_car: CAR filtered data (trials, samples, channels)
    # fs: sampling rate
    # band_definitions: dictionary of band names and (low, high) frequencies
    # ordered_band_names_list: list of band names to ensure consistent feature order
    # Returns: 3D numpy array (trials, channels, bands_ordered)

    n_trials, _, n_channels = data_car.shape
    n_bands = len(ordered_band_names_list)
    
    features_3d = np.zeros((n_trials, n_channels, n_bands))
    
    for t in range(n_trials):
        for ch in range(n_channels):
            signal_one_trial_one_channel = data_car[t, :, ch]
            
            # Compute band powers for this specific signal
            powers = compute_band_power(signal_one_trial_one_channel, fs, band_definitions)
            
            # Store them in the 3D feature array in the specified order
            for b_idx, b_name in enumerate(ordered_band_names_list):
                features_3d[t, ch, b_idx] = powers.get(b_name, 0) # Default to 0 if band missing
                
    return features_3d

#main logic
results_list = []
data_base_path = '/Users/refobic/Downloads/Project/Data/'

for subj in range(1, 6):
    print(f"\nProcessing Subject {subj}...")

    try:     #load training and testing data
        train_mat = loadmat(f'{data_base_path}train_data_{subj}.mat')
        test_mat = loadmat(f'{data_base_path}test_data_{subj}.mat')
    except FileNotFoundError:
        print(f"Error: Data files for subject {subj} not found in {data_base_path}. Skipping.")
        continue

    data_train = train_mat['data']  # trials x samples x channels
    labels_train = train_mat['labels'].squeeze()
    data_test = test_mat['data']
    labels_test = test_mat['labels'].squeeze()
    
    fs = float(train_mat['fs'].squeeze())
    # Handle different structures of 'channels' in .mat files
    if train_mat['channels'].ndim == 1 or train_mat['channels'].shape[0] == 1 or train_mat['channels'].shape[1] == 1:
        channels = [str(c[0]) if isinstance(c, np.ndarray) and c.size > 0 and isinstance(c[0], (np.str_, str))
                    else str(c) if isinstance(c, (np.str_, str))
                    else f"Ch{i+1}" # Fallback name
                    for i, c in enumerate(train_mat['channels'].squeeze())]
    elif train_mat['channels'].ndim == 2: # Often (1, num_channels) with cell arrays
         channels = [str(train_mat['channels'][0, i][0]) if isinstance(train_mat['channels'][0, i], np.ndarray) and train_mat['channels'][0,i].size > 0
                    else str(train_mat['channels'][0, i])
                    for i in range(train_mat['channels'].shape[1])]
    else: # Fallback if unsure
        channels = [f"Ch{i+1}" for i in range(data_train.shape[2])]

    n_channels = data_train.shape[2]
    n_bands_defined = len(band_names_ordered)

    # Step 1: Apply Common Average Reference (CAR) filtering
    data_train_car = apply_car(data_train)
    data_test_car = apply_car(data_test)
    
    # Step 2.1 & 2.2: Extract band power features for all channels and bands
    # This creates a 3D matrix: (trials, channels, bands)
    features_train_3d = extract_features(data_train_car, fs, bands, band_names_ordered)
    features_test_3d = extract_features(data_test_car, fs, bands, band_names_ordered)

    # Step 2.3 & 2.4

    # Scenario 1: Single Channel & Single Band
    print("  Scenario 1: Single Channel & Band")
    for ch_idx, ch_name in enumerate(channels):
        for b_idx, b_name in enumerate(band_names_ordered):
            X_tr = features_train_3d[:, ch_idx, b_idx].reshape(-1, 1)
            X_te = features_test_3d[:, ch_idx, b_idx].reshape(-1, 1)
            
            best_acc_s1, best_k_s1 = 0.0, 1
            for K_val in range(1, 11):
                if K_val > len(X_tr): break #n_neighbors must be less than n_samples
                knn = KNeighborsClassifier(n_neighbors=K_val)
                knn.fit(X_tr, labels_train)
                acc = knn.score(X_te, labels_test)
                if acc > best_acc_s1:
                    best_acc_s1, best_k_s1 = acc, K_val
            results_list.append({
                'Subject': subj, 'Scenario': 'Single Channel & Band',
                'Channel': ch_name, 'Band': b_name,
                'Best K': best_k_s1, 'Accuracy': best_acc_s1
            })

    # Scenario 2: Single Channel & All Bands
    print("  Scenario 2: Single Channel, All Bands")
    for ch_idx, ch_name in enumerate(channels):
        X_tr = features_train_3d[:, ch_idx, :] # Features: (trials, all_bands_for_this_channel)
        X_te = features_test_3d[:, ch_idx, :]
        
        best_acc_s2, best_k_s2 = 0.0, 1
        for K_val in range(1, 11):
            if K_val > len(X_tr): break
            knn = KNeighborsClassifier(n_neighbors=K_val)
            knn.fit(X_tr, labels_train)
            acc = knn.score(X_te, labels_test)
            if acc > best_acc_s2:
                best_acc_s2, best_k_s2 = acc, K_val
        results_list.append({
            'Subject': subj, 'Scenario': 'Single Channel, All Bands',
            'Channel': ch_name, 'Band': 'All',
            'Best K': best_k_s2, 'Accuracy': best_acc_s2
        })

    # Scenario 3: Single Band & All Channels
    print("  Scenario 3: Single Band, All Channels")
    for b_idx, b_name in enumerate(band_names_ordered):
        X_tr = features_train_3d[:, :, b_idx] # Features: (trials, all_channels_for_this_band)
        X_te = features_test_3d[:, :, b_idx]
        
        best_acc_s3, best_k_s3 = 0.0, 1
        for K_val in range(1, 11):
            if K_val > len(X_tr): break
            knn = KNeighborsClassifier(n_neighbors=K_val)
            knn.fit(X_tr, labels_train)
            acc = knn.score(X_te, labels_test)
            if acc > best_acc_s3:
                best_acc_s3, best_k_s3 = acc, K_val
        results_list.append({
            'Subject': subj, 'Scenario': 'Single Band, All Channels',
            'Channel': 'All', 'Band': b_name,
            'Best K': best_k_s3, 'Accuracy': best_acc_s3
        })
        
    # Scenario 4: All Channels & All Bands
    print("  Scenario 4: All Channels & All Bands")
    X_tr = features_train_3d.reshape(features_train_3d.shape[0], n_channels * n_bands_defined)
    X_te = features_test_3d.reshape(features_test_3d.shape[0], n_channels * n_bands_defined)
    
    best_acc_s4, best_k_s4 = 0.0, 1
    for K_val in range(1, 11):
        if K_val > len(X_tr): break
        knn = KNeighborsClassifier(n_neighbors=K_val)
        knn.fit(X_tr, labels_train)
        acc = knn.score(X_te, labels_test)
        if acc > best_acc_s4:
            best_acc_s4, best_k_s4 = acc, K_val
    results_list.append({
        'Subject': subj, 'Scenario': 'All Channels & All Bands',
        'Channel': 'All', 'Band': 'All',
        'Best K': best_k_s4, 'Accuracy': best_acc_s4
    })

# output
df_results = pd.DataFrame(results_list)
output_csv_path = f'{data_base_path}../EEG_KNN_Results_Detailed.csv' # Save one level up from Data folder
df_results.to_csv(output_csv_path, index=False)
print(f"\nDetailed results saved to {output_csv_path}")

# For Deliverable 2 Identify the frequency band and channel and value of K for KNN that gets the highest accuracy on test data for each subject.
best_single_combo_results_list = []
for subj_num in range(1, 6):
    subject_df = df_results[
        (df_results['Subject'] == subj_num) &
        (df_results['Scenario'] == 'Single Channel & Band')
    ]
    if not subject_df.empty:
        best_row = subject_df.loc[subject_df['Accuracy'].idxmax()]
        best_single_combo_results_list.append({
            'Subject': subj_num,
            'Best Channel': best_row['Channel'],
            'Best Band': best_row['Band'],
            'Best K for Combo': best_row['Best K'],
            'Highest Accuracy (Single Combo)': best_row['Accuracy']
        })
    else:
        #this case should not happen if subjects 1-5 are processed and files exist
        best_single_combo_results_list.append({
            'Subject': subj_num, 'Best Channel': 'N/A', 'Best Band': 'N/A',
            'Best K for Combo': 'N/A', 'Highest Accuracy (Single Combo)': 0.0
        })

df_best_single_per_subject = pd.DataFrame(best_single_combo_results_list)
print("\n--- Best Single Channel/Band Performance per Subject (for Deliverable 2) ---")
print(df_best_single_per_subject)

best_single_output_path = f'{data_base_path}../EEG_Best_Single_Combo_Per_Subject.csv'
df_best_single_per_subject.to_csv(best_single_output_path, index=False)
print(f"Best single combo per subject results saved to {best_single_output_path}")

print("\nProcessing Complete.")