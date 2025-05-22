# Ismaiel Sabet 900221277
# Mohamed El-Refai 900222334
# DSP project

# Import necessary libraries
import numpy as np  # For numerical operations, especially with arrays
from scipy.io import loadmat  # For loading data from .mat files (MATLAB format)
from sklearn.neighbors import KNeighborsClassifier  # For K-Nearest Neighbors classification
import pandas as pd  # For data manipulation and analysis, especially using DataFrames
from numpy.fft import rfft, rfftfreq # For computing the Real Fast Fourier Transform and its corresponding frequencies

# Algorithm Outline (as provided by the user)
    # Step 1: Compute the CAR filter to the data
    # Step 2:
        #   2.1: For each electrode of the *7* electrodes,
        #   2.2: For each trial of each class of attention (focused versus drowsy), compute the Fourier Transform and compute the power in the delta, theta, alpha, beta, and gamma
        #   2.3: Apply KNN
        #   2.4: Compute the classification error for each value of K (from 2.3)

# Define frequency bands and their ranges in Hz
bands = {
    'Delta': (0.5, 4),  # Delta waves (0.5-4 Hz)
    'Theta': (4, 8),    # Theta waves (4-8 Hz)
    'Alpha': (8, 13),   # Alpha waves (8-13 Hz)
    'Beta': (13, 30),   # Beta waves (13-30 Hz)
    'Gamma': (30, 45)   # Gamma waves (30-45 Hz)
}

# For consistent ordering of bands when creating feature arrays.
# This ensures that features are always in the same column order.
band_names_ordered = list(bands.keys())

# Function to apply Common Average Reference (CAR) filter
def apply_car(data):
    # CAR filter subtracts the average signal of all channels from each channel.
    # This helps to reduce common noise across electrodes.
    # data shape is expected to be (trials, samples, channels)
    # np.mean(data, axis=2, keepdims=True) calculates the mean across the channels (axis=2).
    # keepdims=True ensures the resulting mean array can be broadcasted for subtraction.
    return data - np.mean(data, axis=2, keepdims=True)

# Function to compute power in specified frequency bands for a single signal trace
def compute_band_power(signal, fs, band_definitions):
    # Args:
    #   signal (np.array): 1D array of time-series EEG data for one trial, one channel.
    #   fs (float): Sampling frequency of the signal.
    #   band_definitions (dict): Dictionary defining band names and their (low_freq, high_freq) tuples.
    # Returns:
    #   dict: A dictionary where keys are band names and values are the mean power in that band.

    n = len(signal)  # Number of samples in the signal
    if n == 0: # Handle cases where an empty signal might be passed
        return {band_name: 0 for band_name in band_definitions} # Return zero power for all bands

    yf = rfft(signal)  # Compute the Real Fast Fourier Transform (RFFT) of the signal.
                       # RFFT is used for real-valued signals, and it50's more efficient.
    power_spectrum = np.abs(yf)**2  # Compute the power spectrum (magnitude squared of FFT coefficients).
    xf = rfftfreq(n, 1/fs)  # Get the corresponding frequencies for the RFFT output.
                           # 1/fs is the sampling interval.

    # Calculate the power in each defined frequency band
    band_powers = {}
    for band_name, (low_freq, high_freq) in band_definitions.items():
        # Create a boolean mask to select frequencies within the current band
        band_mask = (xf >= low_freq) & (xf < high_freq)

        # Calculate mean power in the band if any frequencies fall within the band
        if np.any(band_mask) and len(power_spectrum[band_mask]) > 0:
            # np.mean calculates the average power of the frequencies within the band
            band_powers[band_name] = np.mean(power_spectrum[band_mask])
        else:
            # If no frequencies from the signal fall into this band, assign 0 power
            band_powers[band_name] = 0

    return band_powers

# Function to extract band power features for all trials, channels, and bands
def extract_features(data_car, fs, band_definitions, ordered_band_names_list):
    # Args:
    #   data_car (np.array): CAR filtered EEG data with shape (trials, samples, channels).
    #   fs (float): Sampling rate of the EEG data.
    #   band_definitions (dict): Dictionary defining band names and their frequency ranges.
    #   ordered_band_names_list (list): List of band names to ensure consistent feature order.
    # Returns:
    #   np.array: A 3D numpy array containing features with shape (trials, channels, bands_ordered).

    n_trials, _, n_channels = data_car.shape  # Get dimensions from the input data
    n_bands = len(ordered_band_names_list)    # Get the number of defined frequency bands

    # Initialize a 3D numpy array to store the extracted features
    # Dimensions: (number of trials, number of channels, number of bands)
    features_3d = np.zeros((n_trials, n_channels, n_bands))

    # Iterate over each trial
    for t in range(n_trials):
        # Iterate over each channel in the current trial
        for ch in range(n_channels):
            # Extract the signal for the current trial and channel
            signal_one_trial_one_channel = data_car[t, :, ch]

            # Compute band powers for this specific signal (one trial, one channel)
            powers = compute_band_power(signal_one_trial_one_channel, fs, band_definitions)

            # Store the computed powers in the 3D feature array in the specified order
            for b_idx, b_name in enumerate(ordered_band_names_list):
                # .get(b_name, 0) retrieves the power for b_name, defaulting to 0 if the band is missing (should not happen with current setup)
                features_3d[t, ch, b_idx] = powers.get(b_name, 0)

    return features_3d

# --- Main Script Logic ---
results_list = []  # List to store results from different KNN scenarios for later analysis
data_base_path = '/Users/refobic/Downloads/Project/Data/' # Base path where data files are located

# Loop through each subject (assuming 5 subjects with data files named train_data_1.mat, test_data_1.mat, etc.)
for subj in range(1, 6): # Process subjects 1 through 5
    print(f"\nProcessing Subject {subj}...")

    try:
        # Load training data for the current subject
        train_mat = loadmat(f'{data_base_path}train_data_{subj}.mat')
        # Load testing data for the current subject
        test_mat = loadmat(f'{data_base_path}test_data_{subj}.mat')
    except FileNotFoundError:
        # Handle cases where data files for a subject might be missing
        print(f"Error: Data files for subject {subj} not found in {data_base_path}. Skipping.")
        continue # Skip to the next subject

    # Extract data and labels from the loaded .mat files
    data_train = train_mat['data']  # EEG data for training (trials x samples x channels)
    labels_train = train_mat['labels'].squeeze() # Training labels (e.g., focused vs. drowsy), .squeeze() removes single-dimensional entries
    data_test = test_mat['data']    # EEG data for testing
    labels_test = test_mat['labels'].squeeze()  # Testing labels

    # Extract sampling frequency (fs) and channel names
    fs = float(train_mat['fs'].squeeze()) # Convert sampling frequency to float

    # Handle potentially different structures of 'channels' field in .mat files
    # This section tries to robustly extract channel names.
    if train_mat['channels'].ndim == 1 or train_mat['channels'].shape[0] == 1 or train_mat['channels'].shape[1] == 1:
        # Handles cases where channels are in a simple array or a row/column vector within the .mat file
        channels = [str(c[0]) if isinstance(c, np.ndarray) and c.size > 0 and isinstance(c[0], (np.str_, str))
                    else str(c) if isinstance(c, (np.str_, str))
                    else f"Ch{i+1}" # Fallback naming if extraction fails
                    for i, c in enumerate(train_mat['channels'].squeeze())]
    elif train_mat['channels'].ndim == 2: # Often (1, num_channels) with cell arrays in MATLAB
         # Handles cases where channels are stored in a 2D array, common for cell arrays in MATLAB
         channels = [str(train_mat['channels'][0, i][0]) if isinstance(train_mat['channels'][0, i], np.ndarray) and train_mat['channels'][0,i].size > 0
                    else str(train_mat['channels'][0, i])
                    for i in range(train_mat['channels'].shape[1])]
    else: # Fallback if the channel structure is not recognized
        channels = [f"Ch{i+1}" for i in range(data_train.shape[2])]

    n_channels = data_train.shape[2] # Number of channels from the data dimensions
    n_bands_defined = len(band_names_ordered) # Number of defined frequency bands

    # Step 1: Apply Common Average Reference (CAR) filtering to training and testing data
    data_train_car = apply_car(data_train)
    data_test_car = apply_car(data_test)

    # Step 2.1 & 2.2: Extract band power features for all channels and bands
    # This creates a 3D matrix: (trials, channels, bands)
    features_train_3d = extract_features(data_train_car, fs, bands, band_names_ordered)
    features_test_3d = extract_features(data_test_car, fs, bands, band_names_ordered)

    # Step 2.3 & 2.4: Apply KNN and compute classification accuracy for different feature combinations
    # We test K values from 1 to 10 to find the best K for each scenario.

    # Scenario 1: Using features from a Single Channel & a Single Band
    print("  Scenario 1: Single Channel & Band")
    # Iterate over each channel
    for ch_idx, ch_name in enumerate(channels):
        # Iterate over each frequency band
        for b_idx, b_name in enumerate(band_names_ordered):
            # Extract features for the current channel and band for all trials
            # X_tr will have shape (n_trials, 1) because KNN expects 2D input
            X_tr = features_train_3d[:, ch_idx, b_idx].reshape(-1, 1)
            X_te = features_test_3d[:, ch_idx, b_idx].reshape(-1, 1)

            best_acc_s1, best_k_s1 = 0.0, 1 # Initialize best accuracy and corresponding K
            # Test K values from 1 to 10
            for K_val in range(1, 11):
                if K_val > len(X_tr): break # K cannot be greater than the number of training samples
                knn = KNeighborsClassifier(n_neighbors=K_val) # Initialize KNN classifier
                knn.fit(X_tr, labels_train) # Train the KNN model
                acc = knn.score(X_te, labels_test) # Evaluate accuracy on the test set
                if acc > best_acc_s1: # If current K gives better accuracy
                    best_acc_s1, best_k_s1 = acc, K_val # Update best accuracy and K
            # Store results for this specific channel-band combination
            results_list.append({
                'Subject': subj, 'Scenario': 'Single Channel & Band',
                'Channel': ch_name, 'Band': b_name,
                'Best K': best_k_s1, 'Accuracy': best_acc_s1
            })

    # Scenario 2: Using features from a Single Channel & All Bands
    print("  Scenario 2: Single Channel, All Bands")
    # Iterate over each channel
    for ch_idx, ch_name in enumerate(channels):
        # Extract features for the current channel, using all bands
        # X_tr will have shape (n_trials, n_bands_defined)
        X_tr = features_train_3d[:, ch_idx, :]
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
            'Channel': ch_name, 'Band': 'All', # 'All' indicates all bands are used
            'Best K': best_k_s2, 'Accuracy': best_acc_s2
        })

    # Scenario 3: Using features from a Single Band & All Channels
    print("  Scenario 3: Single Band, All Channels")
    # Iterate over each frequency band
    for b_idx, b_name in enumerate(band_names_ordered):
        # Extract features for the current band, using all channels
        # X_tr will have shape (n_trials, n_channels)
        X_tr = features_train_3d[:, :, b_idx]
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
            'Channel': 'All', # 'All' indicates all channels are used
            'Band': b_name,
            'Best K': best_k_s3, 'Accuracy': best_acc_s3
        })

    # Scenario 4: Using features from All Channels & All Bands
    print("  Scenario 4: All Channels & All Bands")
    # Reshape the 3D feature matrix into a 2D matrix for KNN: (trials, channels * bands)
    # This concatenates features from all channels and all bands for each trial.
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

# --- Output and Further Analysis ---

# Convert the list of result dictionaries into a Pandas DataFrame for easier analysis and saving
df_results = pd.DataFrame(results_list)
# Define the path to save the detailed results CSV file
# Saves it one directory level up from the 'Data' folder
output_csv_path = f'{data_base_path}../EEG_KNN_Results_Detailed.csv'
df_results.to_csv(output_csv_path, index=False) # Save DataFrame to CSV, without writing row indices
print(f"\nDetailed results saved to {output_csv_path}")

# For Deliverable 2: Identify the frequency band and channel and value of K for KNN
# that gets the highest accuracy on test data for each subject from Scenario 1 (Single Channel & Single Band).
best_single_combo_results_list = [] # List to store the best single combination for each subject
# Iterate through each subject again to find their best result from Scenario 1
for subj_num in range(1, 6):
    # Filter the DataFrame for the current subject and for "Single Channel & Band" scenario
    subject_df = df_results[
        (df_results['Subject'] == subj_num) &
        (df_results['Scenario'] == 'Single Channel & Band')
    ]
    if not subject_df.empty: # If results exist for this subject and scenario
        # Find the row with the maximum accuracy for this subject in Scenario 1
        best_row = subject_df.loc[subject_df['Accuracy'].idxmax()]
        # Append the details of this best combination to the list
        best_single_combo_results_list.append({
            'Subject': subj_num,
            'Best Channel': best_row['Channel'],
            'Best Band': best_row['Band'],
            'Best K for Combo': best_row['Best K'],
            'Highest Accuracy (Single Combo)': best_row['Accuracy']
        })
    else:
        # This case should ideally not happen if subjects 1-5 are processed and files exist.
        # Handles missing data gracefully.
        best_single_combo_results_list.append({
            'Subject': subj_num, 'Best Channel': 'N/A', 'Best Band': 'N/A',
            'Best K for Combo': 'N/A', 'Highest Accuracy (Single Combo)': 0.0
        })

# Convert the list of best single combinations per subject into a DataFrame
df_best_single_per_subject = pd.DataFrame(best_single_combo_results_list)
print("\n--- Best Single Channel/Band Performance per Subject (for Deliverable 2) ---")
print(df_best_single_per_subject) # Display the summary table

# Define the path and save the summary table for the best single channel/band per subject
best_single_output_path = f'{data_base_path}../EEG_Best_Single_Combo_Per_Subject.csv'
df_best_single_per_subject.to_csv(best_single_output_path, index=False)
print(f"Best single combo per subject results saved to {best_single_output_path}")

print("\nProcessing Complete.")
