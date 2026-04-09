# Pipeline Klasör Yapısı

Bu klasör, TÜBİTAK 2209-A projesinin tüm makine öğrenmesi pipeline'ını içerir.
ml-aaaa klasörüne dokunulmaz — burası tamamen ayrı ve temizdir.

---

## Genel Akış

```
Ham Veri
   │
   ▼
veri/islenmis/          → Train / Validation / Test ayrımı yapılmış veriler
   │
   ├──▶ asama1_feature_tahmini/
   │         Nüfus, GDP, yenilenebilir enerji oranı gibi
   │         elektriği etkileyen parametrelerin 2021-2030
   │         tahmini yapılır.
   │         Çıktı: her feature için 2021-2030 tahmin değerleri
   │
   └──▶ asama2_elektrik_tahmini/
             Geçmiş elektrik tüketimi + geçmiş feature değerleri
             + Aşama 1'den gelen 2026-2030 feature tahminleri
             kullanılarak 2026-2030 elektrik tüketimi tahmin edilir.
             Çıktı: 2026-2030 Türkiye elektrik tüketim tahmini
                    → karbon ayak izi hesabı

```

---

## Klasörler

### veri/
| Alt Klasör | Ne Var | Açıklama |
|---|---|---|
| ham/ | merged_clean.csv | Kaynak veri — hiç değiştirilmez |
| islenmis/ | turkiye_train.csv, turkiye_val.csv, turkiye_test.csv | Split edilmiş veriler |

**Split:**
- Train      : 1990 – 2020 (model bu yıllarda öğrenir)
- Validation : 2021 – 2023 (hiperparametre ayarı için)
- Test       : 2024 – 2025 (son değerlendirme)
- Forecast   : 2026 – 2030 (asıl hedef)

---

### asama1_feature_tahmini/
Elektriği etkileyen parametrelerin gelecek değerlerini tahmin eder.
Bu tahminler Aşama 2'ye feature olarak verilir.

| Alt Klasör | Ne Var | Açıklama |
|---|---|---|
| kodlar/ | asama1_veri_hazirla.py, asama1_egit.py | Train ve tahmin kodları |
| modeller/ | feature_model_nufus.pkl, feature_model_gdp.pkl ... | Her feature için ayrı kayıtlı model |
| ciktilar/ | feature_tahminleri_2021_2030.csv | Tüm feature'ların 2021-2030 tahminleri |

**Tahmin edilen feature'lar:**
- Population (Nüfus)
- GDP_Per_Capita (Kişi başına gelir)
- GDP_Growth (Büyüme oranı)
- renewables_share_elec (Yenilenebilir enerji oranı)
- Urbanization_Rate (Kentleşme oranı)
- Industry_Value_Added_Pct_GDP (Sanayi katma değeri)
- Population_Density (Nüfus yoğunluğu)

---

### asama2_elektrik_tahmini/
Elektrik tüketimini tahmin eden ana model.

| Alt Klasör | Ne Var | Açıklama |
|---|---|---|
| kodlar/ | asama2_veri_hazirla.py, asama2_egit.py | Train ve tahmin kodları |
| modeller/ | lstm_model/, rf_model.pkl, xgboost_model.pkl | 3 algoritmanın modelleri |
| ciktilar/ | elektrik_tahmin_2026_2030.csv, model_metrikleri.csv | Tahminler ve RMSE/MAPE/R² skorları |

**Girdi (2026-2030 tahmini için):**
- Geçmiş electricity_demand serisi (1990–2025, gerçek)
- Geçmiş feature değerleri (1990–2025, gerçek)
- 2026–2030 feature değerleri ← Aşama 1'den gelir

**Çıktı:** 2026, 2027, 2028, 2029, 2030 elektrik tüketim tahmini (TWh)

---

### sonuclar/
| Alt Klasör | Ne Var | Açıklama |
|---|---|---|
| tahminler/ | final_elektrik_2026_2030.csv, karbon_ayak_izi_2026_2030.csv | Nihai çıktılar |
| gorseller/ | *.png grafik dosyaları | Raporlara girecek görseller |

---

## Çalıştırma Sırası

```
1. python veri/islenmis/split_yap.py
2. python asama1_feature_tahmini/kodlar/asama1_egit.py
3. python asama2_elektrik_tahmini/kodlar/asama2_egit.py
```
