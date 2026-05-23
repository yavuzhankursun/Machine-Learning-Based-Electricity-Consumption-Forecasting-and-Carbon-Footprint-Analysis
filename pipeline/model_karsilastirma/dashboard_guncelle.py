"""
DASHBOARD GUNCELLE — 7 model karsilastirma sonuclarini data.json'a enjekte eder
================================================================================
pipeline/sonuclar/model_karsilastirma/asama{1,2,3}_model_karsilastirma.json
dosyalarini okur, dashboard/public/data.json icine `model_karsilastirma`
anahtari olarak ekler (mevcut anahtarlara DOKUNMAZ).

Kullanim:
  python3 dashboard_guncelle.py
Sonra dashboard'da `npm run build` calistirilir (public → dist kopyalanir).
"""

import os, json, sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR     = os.path.dirname(PIPELINE_DIR)
KARS_DIR     = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
DATA_JSON    = os.path.join(ROOT_DIR, 'dashboard', 'public', 'data.json')

MODEL_SIRASI = ['RF', 'XGBoost', 'LightGBM', 'CatBoost', 'LSTM', 'ARIMA', 'SARIMA']
METRIK_ANAHTARLARI = ('r2', 'mae', 'rmse', 'mape', 'cv_n', 'order', 'seasonal_order')


def _yukle(ad):
    yol = os.path.join(KARS_DIR, ad)
    if not os.path.exists(yol):
        return None
    with open(yol) as f:
        return json.load(f)


def _temiz_metrik(md):
    """Sadece arayuzun kullandigi metrik alanlarini birak."""
    if not md:
        return {}
    return {k: md.get(k) for k in METRIK_ANAHTARLARI if k in md}


def main():
    if not os.path.exists(DATA_JSON):
        print(f"HATA: {DATA_JSON} bulunamadi"); sys.exit(1)

    a1 = _yukle('asama1_model_karsilastirma.json')
    a2 = _yukle('asama2_model_karsilastirma.json')
    a3 = _yukle('asama3_model_karsilastirma.json')

    if not (a1 or a2 or a3):
        print("HATA: hicbir karsilastirma JSON'u bulunamadi. Once egitimleri calistir.")
        sys.exit(1)

    mk = {'modeller': MODEL_SIRASI}

    # ── Asama 1: feature x model matrisi ──
    if a1:
        satirlar = []
        for feat, d in a1.items():
            satirlar.append({
                'feature': feat,
                'en_iyi': d.get('en_iyi'),
                'modeller': {m: _temiz_metrik(d.get('modeller', {}).get(m, {}))
                             for m in MODEL_SIRASI},
            })
        mk['asama1'] = {'satirlar': satirlar}
        print(f"  Asama 1: {len(satirlar)} feature x {len(MODEL_SIRASI)} model")

    # ── Asama 2 / 3: tek hedef ──
    for ad, src in [('asama2', a2), ('asama3', a3)]:
        if not src:
            continue
        mk[ad] = {
            'hedef': src.get('hedef'),
            'birim': src.get('birim'),
            'en_iyi': src.get('en_iyi'),
            'n_obs': src.get('n_obs'),
            'min_train': src.get('min_train'),
            'modeller': {m: _temiz_metrik(src.get('modeller', {}).get(m, {}))
                         for m in MODEL_SIRASI},
        }
        ei = src.get('en_iyi')
        r2 = src.get('modeller', {}).get(ei, {}).get('r2') if ei else None
        print(f"  {ad}: en iyi {ei} (R²={r2})")

    # ── data.json'a enjekte et (mevcut anahtarlari koru) ──
    with open(DATA_JSON) as f:
        data = json.load(f)
    data['model_karsilastirma'] = mk

    tmp = DATA_JSON + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, DATA_JSON)

    print(f"\n  ✓ {DATA_JSON} guncellendi (model_karsilastirma eklendi)")
    print(f"  Sonraki adim: cd dashboard && npm run build")


if __name__ == '__main__':
    main()
