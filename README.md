# Tolstoy Tarzı Metin Üretimi: LSTM ile Üretken Dil Modeli
Lev Tolstoy'un edebi eserlerinden öğrenen karakter-seviyesi üretken dil modeli

## 🎯 Proje Hakkında
Bu proje, Lev Tolstoy'un klasik eserlerinden (Anna Karenina, War and Peace) öğrenerek, yazarın üslubunda metin üretebilen karakter-seviyesi derin öğrenme modeli geliştirmeyi amaçlamaktadır.

### Temel Özellikler

✅ LSTM tabanlı derin öğrenme mimarisi  
✅ Karakter-seviyesi metin üretimi  
✅ Temperature kontrolü ile yaratıcılık ayarı  
✅ İnteraktif Streamlit demo arayüzü  
✅ ~2.7 milyon karakterlik eğitim verisi

### Teknik Detaylar

| Özellik | Değer |
|---------|-------|
| Model | Bidirectional LSTM |
| Vocabulary | 45 karakter |
| Parametre Sayısı | ~343K |
| Validation Loss | 1.3380 |
| Perplexity | 3.81 |

📄 **Detaylı teorik altyapı, mimari açıklamaları ve sonuçlar için report.pdf dosyasına bakınız.**

---

## 🚀 Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

**Ana bağımlılıklar:**
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Streamlit (demo için)

---

## 💻 Kullanım
#### NOT :
##### Öncelikle sequences.zip dosyasından sequences.pkl dosyasını çıkarın 

### 1. Yeni Veri Seti Ekleme

Kendi metinlerinizle model eğitmek için:

**Adım 1: Veri Hazırlama**
```bash
# Yeni metin dosyasını data/ klasörüne ekleyin
# Örnek: data/yeni_eser.txt
```

**Adım 2: Veri Ön İşleme**
```bash
python src/preprocess.py \
  --input_files data/anna_karenina.txt data/war_and_peace.txt data/yeni_eser.txt \
  --output_dir artifacts/ \
  --seq_length 100
```

**Parametreler:**
- `--input_files`: Eğitimde kullanılacak metin dosyaları (boşlukla ayrılmış)
- `--output_dir`: Ön işlenmiş verilerin kaydedileceği klasör
- `--seq_length`: Karakter dizisi uzunluğu (varsayılan: 100)

**Adım 3: Model Eğitimi**

Google Colab'da `train_colab.ipynb` notebook'unu açın ve çalıştırın:
- Artifacts klasöründeki yeni ön işlenmiş verileri kullanacaktır
- Eğitim tamamlandığında yeni model artifacts/ klasörüne kaydedilir

**Dosya Formatı Gereksinimleri:**
- ✅ Düz metin (.txt) formatı
- ✅ UTF-8 encoding
- ✅ Minimum 100KB boyut (önerilen: >1MB)
- ❌ Özel karakterler, emoji'ler temizlenmelidir

**Örnek Kullanım:**
```bash
# Tek dosya ile
python src/preprocess.py --input_files data/yeni_eser.txt

# Birden fazla dosya ile
python src/preprocess.py \
  --input_files data/dosya1.txt data/dosya2.txt data/dosya3.txt \
  --seq_length 150
```

**Çıktı:**
```
✓ Toplam karakter: 3,245,678
✓ Vocabulary boyutu: 52
✓ Eğitim dizisi: 32,456
✓ Kaydedilen dosyalar:
  - artifacts/sequences.npy
  - artifacts/char_to_idx.json
  - artifacts/idx_to_char.json
  - artifacts/preprocessing_summary.json
```
### 2. Metin Üretimi (Komut Satırı)
```bash
python generate.py \
  --seed "the old man looked at" \
  --length 400 \
  --temperature 0.5
```

**Parametreler:**
- `--seed`: Başlangıç metni
- `--length`: Üretilecek karakter sayısı
- `--temperature`: Yaratıcılık seviyesi (0.2-1.5)
  - **0.2**: Tutarlı, güvenli
  - **0.5**: Dengeli ⭐ (önerilen)
  - **1.0**: Yaratıcı, riskli

**Örnek Çıktı:**
```
Seed: "the old man looked at"
Temperature: 0.5

Generated text:
the old man looked at the window and saw the children 
playing in the garden. He remembered his youth, when 
he was happy and free...
```

### 3. İnteraktif Demo (Streamlit)

```bash
python -m streamlit run app.py
```

**Demo Özellikleri:**
- 🎨 Seed text girişi
- 🌡️ Temperature slider kontrolü
- 📏 Uzunluk ayarı
- 🔄 Gerçek zamanlı üretim
- 📊 Farklı temperature'ları karşılaştırma

**Demo Ekran Görüntüsü:**

<img width="1909" height="974" alt="image" src="https://github.com/user-attachments/assets/18fc94fa-00bc-4b17-a83b-91e074d5e206" />

---

## 📁 Proje Yapısı

```
project/
├── data/                          # Veri 
│   ├── anna_karenina.txt
│   └── war_and_peace.txt
│
├── artifacts/                     # Model çıktıları
│   ├── best_model.keras          # Eğitilmiş 
│   ├── char_to_idx.json          # Karakter → 
│   ├── idx_to_char.json          # İndeks → 
│   └── preprocessing_summary.json
│
├── src/                           # Kaynak kodlar
│   └── preprocess.py             # Veri ön işleme
│
├── train_colab.ipynb             # Colab eğitim 
├── generate.py                    # Metin üretimi
├── app.py                         # Streamlit 
└── README.md                      # Bu dosya
```

---

## 📊 Hızlı Sonuçlar

### Model Performansı

- **Validation Loss:** 1.3380
- **Perplexity:** 3.81 (karakter-seviyesi için mükemmel)
- **Eğitim Süresi:** ~4-6 saat (Tesla T4 GPU)

### Temperature Karşılaştırması

| Temperature | Tutarlılık | Yaratıcılık | Kullanım |
|-------------|------------|-------------|----------|
| 0.2 | ⭐⭐⭐⭐⭐ | ⭐⭐ | Güvenli üretim |
| 0.5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Genel kullanım ⭐ |
| 1.0 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Deneysel |

---

## 📖 Detaylı Dokümantasyon

Bu README temel kullanım bilgilerini içermektedir. Detaylı bilgi için:

📄 **report.md** - Kapsamlı proje raporu:
- Teorik altyapı (Generative vs Discriminative modeller)
- Tarihsel gelişim (Markov → RNN → LSTM → Transformer)
- Detaylı model mimarisi ve tasarım kararları
- Eğitim süreci ve hiperparametre seçimleri
- Kapsamlı değerlendirme ve sonuçlar
- Akademik kaynaklar ve referanslar

---

## 🔬 Teknik Highlights

### Neden LSTM?
- Orta ölçekli veri seti için optimal (~2.7M karakter)
- Transformer'a göre daha az kaynak gereksinimi
- Kanıtlanmış sekans modelleme başarısı

### Neden Karakter-Seviyesi?
- Küçük vocabulary (45 vs binlerce kelime)
- OOV (Out-of-Vocabulary) problemi yok
- Yazım stili ve noktalama öğrenimi

### Neden Bidirectional?
- Hem önceki hem sonraki karakterlerden bağlam
- %10-15 daha iyi performans
- Daha zengin özellik öğrenimi

---

## ⚠️ Bilinen Kısıtlamalar

❌ **Karakter-seviyesi yaklaşım:** Uzun metinlerde anlamsal tutarlılık zorluğu  
❌ **100-200 karakter bağlam penceresi sınırlaması**  
❌ **Kelime-bazlı modellere göre daha yavaş üretim**

✅ **Başarıyla Öğrenilen:** Noktalama, kelime uzunluğu, cümle ritmi, yazım stili

---

## 📚 Kaynaklar

### Temel Referanslar:
- Hochreiter & Schmidhuber (1997) - LSTM orijinal makalesi
- Graves (2013) - Sequence Generation with RNNs
- Karpathy - The Unreasonable Effectiveness of RNNs

**Tüm kaynaklar için:** report.md - Kaynaklar bölümü

**Not:** Bu proje, üretken yapay zeka sistemlerinin temel çalışma mantığını anlamak ve uçtan uca bir dil modeli geliştirme sürecini deneyimlemek amacıyla hazırlanmıştır.

