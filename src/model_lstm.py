"""
LSTM Deep Learning Model
- Tum veri uzerinde sequence olustur, target yilina gore split yap
- Boylece test tahmini icin egitim verisi de sequence'a dahil olur
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import MinMaxScaler
from model_rf import evaluate


def train_lstm(data):
    """LSTM egit ve degerlendir."""
    print("\n=== LSTM (Deep Learning) ===")

    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    np.random.seed(42)

    seq_length = 5
    full_df = data['full_df'].sort_values(['iso_code', 'year']).copy()
    feature_cols = data['feature_cols']

    # --- Tum veriyi olceklendir ---
    X_all = full_df[feature_cols].values.astype(float)
    y_all = full_df['electricity_demand'].values.astype(float)
    years_all = full_df['year'].values
    countries_all = full_df['iso_code'].values

    # NaN temizle
    X_all = np.nan_to_num(X_all, 0)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_all)
    y_scaled = scaler_y.fit_transform(y_all.reshape(-1, 1)).ravel()

    # --- Ulke bazinda sequence olustur, target yilina gore etiketle ---
    all_X_seq = []
    all_y_seq = []
    all_years_seq = []
    all_countries_seq = []

    for country in np.unique(countries_all):
        mask = countries_all == country
        X_c = X_scaled[mask]
        y_c = y_scaled[mask]
        years_c = years_all[mask]

        if len(X_c) < seq_length:
            continue

        for i in range(seq_length, len(X_c) + 1):
            X_seq = X_c[i - seq_length:i]
            y_target = y_c[i - 1]       # Son adimin degeri = tahmin hedefi
            year_target = years_c[i - 1]

            all_X_seq.append(X_seq)
            all_y_seq.append(y_target)
            all_years_seq.append(year_target)
            all_countries_seq.append(country)

    all_X_seq = np.array(all_X_seq)
    all_y_seq = np.array(all_y_seq)
    all_years_seq = np.array(all_years_seq)
    all_countries_seq = np.array(all_countries_seq)

    # --- Temporal split (target yilina gore) ---
    train_mask = all_years_seq <= 2018
    val_mask = (all_years_seq >= 2019) & (all_years_seq <= 2020)
    test_mask = all_years_seq >= 2021

    X_train_seq = all_X_seq[train_mask]
    y_train_seq = all_y_seq[train_mask]
    X_val_seq = all_X_seq[val_mask]
    y_val_seq = all_y_seq[val_mask]
    X_test_seq = all_X_seq[test_mask]
    y_test_seq = all_y_seq[test_mask]

    print(f"  Sequence shapes - Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")

    if len(X_train_seq) == 0:
        print("  UYARI: Yeterli sequence olusturulamadi!")
        return None

    # --- Model ---
    n_features = X_train_seq.shape[2]

    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True,
                         input_shape=(seq_length, n_features)),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    best_epoch = len(history.history['loss'])
    if 'val_loss' in history.history:
        best_val_loss = min(history.history['val_loss'])
        best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    print(f"  Best epoch: {best_epoch}")

    # --- Tahmin ---
    pred_train_scaled = model.predict(X_train_seq, verbose=0).ravel()
    pred_val_scaled = model.predict(X_val_seq, verbose=0).ravel()
    pred_test_scaled = model.predict(X_test_seq, verbose=0).ravel()

    pred_train = scaler_y.inverse_transform(pred_train_scaled.reshape(-1, 1)).ravel()
    pred_val = scaler_y.inverse_transform(pred_val_scaled.reshape(-1, 1)).ravel()
    pred_test = scaler_y.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()

    y_train_orig = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).ravel()
    y_val_orig = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()

    # Negatif tahminleri 0 yap
    pred_train = np.maximum(pred_train, 0)
    pred_val = np.maximum(pred_val, 0)
    pred_test = np.maximum(pred_test, 0)

    # Degerlendirme
    metrics_train = evaluate(y_train_orig, pred_train, "Train")
    metrics_val = evaluate(y_val_orig, pred_val, "Val")
    metrics_test = evaluate(y_test_orig, pred_test, "Test")

    # --- main.py'deki y_test ile karsilastirma icin eşle ---
    # LSTM'in test tahmini farkli uzunlukta olabilir (sequence-based)
    # Orijinal y_test ile ayni boyutta olmalari icin ulke-yil bazinda esle
    test_countries = all_countries_seq[test_mask]
    test_years = all_years_seq[test_mask]

    # Orijinal test_df ile esle
    orig_test = data['test_df']
    pred_test_aligned = np.zeros(len(orig_test))
    y_test_aligned = data['y_test']

    for i, (_, row) in enumerate(orig_test.iterrows()):
        country = row['iso_code']
        year = row['year']
        match = (test_countries == country) & (test_years == year)
        if match.any():
            idx = np.where(match)[0][0]
            pred_test_aligned[i] = pred_test[idx]
        else:
            # Sequence olusturulamayan noktalar icin lag_1 kullan
            pred_test_aligned[i] = row.get('demand_lag_1', data['y_test'][i])

    # Ayni sekilde train ve val icin
    pred_train_aligned = np.zeros(len(data['train_df']))
    train_countries = all_countries_seq[train_mask]
    train_years = all_years_seq[train_mask]
    for i, (_, row) in enumerate(data['train_df'].iterrows()):
        match = (train_countries == row['iso_code']) & (train_years == row['year'])
        if match.any():
            pred_train_aligned[i] = pred_train[np.where(match)[0][0]]
        else:
            pred_train_aligned[i] = data['y_train'][i]

    pred_val_aligned = np.zeros(len(data['val_df']))
    val_countries = all_countries_seq[val_mask]
    val_years = all_years_seq[val_mask]
    for i, (_, row) in enumerate(data['val_df'].iterrows()):
        match = (val_countries == row['iso_code']) & (val_years == row['year'])
        if match.any():
            pred_val_aligned[i] = pred_val[np.where(match)[0][0]]
        else:
            pred_val_aligned[i] = data['y_val'][i]

    # Aligned degerlendirme
    print("\n  Aligned (orijinal y_test ile):")
    metrics_test_a = evaluate(data['y_test'], pred_test_aligned, "Test (aligned)")

    return {
        'model': model,
        'name': 'LSTM',
        'metrics_train': evaluate(data['y_train'], pred_train_aligned, ""),
        'metrics_val': evaluate(data['y_val'], pred_val_aligned, ""),
        'metrics_test': metrics_test_a,
        'pred_train': pred_train_aligned,
        'pred_val': pred_val_aligned,
        'pred_test': pred_test_aligned,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_importance': None,
    }


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_merge import load_and_merge
    from feature_engineering import prepare_data

    df = load_and_merge()
    data = prepare_data(df)
    result = train_lstm(data)
