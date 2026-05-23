# Model Karşılaştırma — 7 Model × 3 Aşama

Üç aşamada da **7 modeli** aynı **walk-forward (genişleyen pencere) çapraz
doğrulama** noktalarında değerlendirir; böylece R² değerleri doğrudan
karşılaştırılabilir.

## Modeller
1. **Random Forest** — feature + lag tabanlı
2. **XGBoost** — feature + lag tabanlı
3. **LightGBM** — feature + lag tabanlı
4. **CatBoost** — feature + lag tabanlı
5. **LSTM** — feature + lag tabanlı (derin öğrenme)
6. **ARIMA** — univariate (hedefin kendi geçmişi)
7. **SARIMA** — univariate mevsimsel baseline (yıllık veride D=0)

> ARIMA/SARIMA feature kullanmaz; tezde standart univariate baseline'dır.
> Yıllık veride mevsimsel fark alma (D=1) kısa seride ıraksadığı için SARIMA
> adayları D=0 ile sınırlandı, tahminler tarihsel banda (±1.5×aralık) kısıtlandı.

## Çalıştırma
```bash
cd pipeline/model_karsilastirma
python3 asama1_karsilastirma.py   # Türkiye, 19 feature (~25 dk, resume edilebilir)
python3 asama2_karsilastirma.py   # Türkiye elektrik talebi (~2 dk)
python3 asama3_karsilastirma.py   # Türkiye karbon ayak izi (~2 dk)
python3 dashboard_guncelle.py     # sonuçları dashboard/public/data.json'a enjekte et
cd ../../dashboard && npm run build
```

## Çıktılar
`pipeline/sonuclar/model_karsilastirma/`:
- `asama1_model_karsilastirma.{csv,json}` — feature × model R²/MAE/RMSE/MAPE
- `asama2_model_karsilastirma.{csv,json}` — model × metrik (elektrik)
- `asama3_model_karsilastirma.{csv,json}` — model × metrik (karbon)

## Dosyalar
- `karsilastirma_ortak.py` — ortak harness (7 model + walk-forward CV + metrikler)
- `asama{1,2,3}_karsilastirma.py` — aşama scriptleri
- `dashboard_guncelle.py` — data.json enjeksiyonu

## Notlar
- Mevcut `asama{1,2,3}_egit.py` ve forecast çıktılarına **dokunmaz**.
- LEAKAGE GUARD: Aşama 3'te co2 türevi feature'lar (coal_co2, gas_co2,
  co2_per_capita...) engellenir; her fold'da scaler yalnız train'e fit edilir.
- Aşama 1 checkpoint'lidir: yarıda kesilirse tamamlanan feature'lar atlanır.
