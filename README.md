# Sıcaklık Anomali Tahmini ve Şehir Kümeleme

Bu projede **GlobalLandTemperaturesByMajorCity** veri seti kullanılarak:
- Şehir-bazlı aylık **climatology** (mevsimsellik) çıkarılır
- Aylık **sıcaklık anomalisi** hesaplanır
- Lag özellikleri + zamansal özelliklerle **Ridge Regression** ile anomali tahmini yapılır
- Seçilen bir yıl/ay için şehirler **K-Means** ile (enlem, boylam, sıcaklık) üzerinden kümelenir
- Çeşitli görseller (MAE trend, residual, scatter, confusion matrix, elbow) üretilir

> Not: Bu repo **en iyi çalışan yaklaşımı** (Ridge + lag’ler + KMeans) içerir. Diğer denemeler (farklı modeller/variantlar) sonuçları iyi olmadığı için repoya alınmadı.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

Repo kök dizininde:

```bash
python src/veri_madeni.py
```

Çalıştırınca şu çıktılar oluşur:
- `plots/` klasörü altında grafikler
- `temperature_anomaly_ridge.joblib`, `city_month_climatology.joblib` model artefact'ları
- `backtest_report.csv` ve `clusters_YYYY_M.csv` gibi rapor çıktıları

## Veri

Bu repoda `data/GlobalLandTemperaturesByMajorCity.csv` dosyası hazır gelir.

Eğer daha büyük olan **GlobalLandTemperaturesByCity.csv** ile koşmak istersen:
- dosyayı `data/` içine koy
- `src/veri_madeni.py` içindeki dosya yolunu değiştirmen yeterli.

## İçerik

- `src/veri_madeni.py` : uçtan uca pipeline (preprocess → model → backtest → plot → clustering)
- `reports/` : sunum ve rapor
- `data/` : veri setleri

