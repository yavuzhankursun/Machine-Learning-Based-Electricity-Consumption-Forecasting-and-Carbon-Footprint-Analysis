"""
LSTM Modeli

KAVRAMLAR:
  epochs=200
    Egitim verisinin kac kez tamamen taranacagi.
    EarlyStopping(patience=20): val_loss 20 epoch boyunca iyilesmezse
    otomatik durulur ve en iyi agirliklar geri yuklenir.
    Genellikle ~60-100. epochta durur.

  batch_size=32
    Her agirlik guncellenmesinde kullanilan ornek sayisi.
    3450 train / 32 = ~108 guncelleme/epoch.
    32: kucuk gurultulu ama iyi genelliyor; 128 olursa daha stabil ama
    overfitting artar.

  LSTM mimarisi:
    Sequence input : [lag_3, lag_2, lag_1] → sekans uzunlugu 3, 1 ozellik
    Auxiliary input: [rolling_mean, rolling_std, yoy_growth, year, country_enc]

    seq_input → LSTM(32) → zamansal pattern
    aux_input → dogrudan
    [LSTM cikti + aux] → Dense(32, relu) → Dropout(0.2) → Dense(1)

    Neden iki giris?
    LSTM'e sekans yerine yardimci ozellikleri de vermek icin.
    Sekans kisminda zamansal kaliplar, aux kisminda istatistiksel ozetler.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from metrics import compute_metrics

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_OK = True
except ImportError:
    TF_OK = False


def split_inputs(X: np.ndarray):
    """
    Feature array'ini ikiye bol:
      X_seq: lag_3, lag_2, lag_1  → shape (n, 3, 1)  LSTM sekans
      X_aux: diger feature'lar    → shape (n, 5)      yardimci giris
    """
    X_seq = X[:, :3].reshape(-1, 3, 1).astype(float)   # lag_3, lag_2, lag_1
    X_aux = X[:, 3:].astype(float)                      # rolling_mean, std, yoy, year, country
    return X_seq, X_aux


def _build_model(n_aux: int) -> 'Model':
    seq_in  = Input(shape=(3, 1),    name='seq_input')
    aux_in  = Input(shape=(n_aux,),  name='aux_input')

    lstm_out = LSTM(32, name='lstm')(seq_in)
    merged   = Concatenate()([lstm_out, aux_in])
    x        = Dense(32, activation='relu')(merged)
    x        = Dropout(0.2)(x)
    out      = Dense(1, name='output')(x)

    model = Model(inputs=[seq_in, aux_in], outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model


def train_lstm(data, epochs: int = 200, batch_size: int = 32):
    """
    epochs=200    : Maksimum 200 epoch, early stopping ile erkenden durabilir.
    batch_size=32 : Her guncelleme 32 ornek.
    """
    if not TF_OK:
        print("TensorFlow yuklu degil, LSTM atlaniyor.")
        return None

    print(f"\n=== LSTM (max_epochs={epochs}, batch={batch_size}, early_stop=20) ===")

    Xtr_s, Xtr_a = split_inputs(data['X_train'])
    Xva_s, Xva_a = split_inputs(data['X_val'])
    Xte_s, Xte_a = split_inputs(data['X_test'])

    model = _build_model(n_aux=Xtr_a.shape[1])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5),
    ]

    hist = model.fit(
        [Xtr_s, Xtr_a], data['y_train'],
        validation_data=([Xva_s, Xva_a], data['y_val']),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    used = len(hist.history['loss'])
    print(f"  Early stopping: {used}/{epochs} epoch kullanildi")
    model.summary(print_fn=lambda x: None)   # sessiz

    def predict(Xs, Xa):
        return model.predict([Xs, Xa], verbose=0).flatten()

    pred_train = predict(Xtr_s, Xtr_a)
    pred_val   = predict(Xva_s, Xva_a)
    pred_test  = predict(Xte_s, Xte_a)

    m_train = compute_metrics(data['y_train'], pred_train, 'Train')
    m_val   = compute_metrics(data['y_val'],   pred_val,   'Val  ')
    m_test  = compute_metrics(data['y_test'],  pred_test,  'Test ')

    return {
        'name'         : 'LSTM',
        'model'        : model,
        'pred_train'   : pred_train,
        'pred_val'     : pred_val,
        'pred_test'    : pred_test,
        'metrics_train': m_train,
        'metrics_val'  : m_val,
        'metrics_test' : m_test,
        'history'      : hist.history,
    }
