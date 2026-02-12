"""
Streamlit Web Application for Tolstoy Style Text Generation
============================================================
Bu Streamlit uygulaması, eğitilmiş LSTM modelini kullanarak
Tolstoy stilinde metin üretimi yapar.

Özellikler:
- Interaktif web arayüzü
- Temperature kontrolü ile yaratıcılık ayarı
- Seed text ile özelleştirilebilir başlangıç
- Gerçek zamanlı metin üretimi
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

# TensorFlow uyarılarını kapat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model, Model

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================
class Config:
    """Uygulama yapılandırma ayarları"""
    SCRIPT_DIR = Path(__file__).parent
    ARTIFACTS_DIR = (SCRIPT_DIR / "artifacts").resolve()
    MODEL_PATH = ARTIFACTS_DIR / "best_model.keras"
    SEQ_LENGTH = 40
    
    # UI Varsayılan değerleri
    DEFAULT_SEED = "the old man looked at"
    DEFAULT_TEMPERATURE = 0.5
    DEFAULT_LENGTH = 400
    MIN_TEMPERATURE = 0.1
    MAX_TEMPERATURE = 1.2
    MIN_LENGTH = 100
    MAX_LENGTH = 800


# =========================
# TEXT GENERATOR CLASS
# =========================
class StreamlitTextGenerator:
    """Streamlit için optimize edilmiş metin üretici sınıfı"""
    
    def __init__(self, artifacts_dir: Path, seq_length: int = 40):
        """
        Args:
            artifacts_dir: Model ve vocabulary dosyalarının bulunduğu dizin
            seq_length: Girdi sequence uzunluğu
        """
        self.artifacts_dir = artifacts_dir
        self.seq_length = seq_length
        
        self.model: Model = None
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def load_model_and_vocab(self) -> None:
        """Model ve vocabulary dosyalarını yükler"""
        try:
            # Model yükleme
            model_path = self.artifacts_dir / "best_model.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            
            logger.info("Model yükleniyor...")
            self.model = load_model(model_path)
            logger.info("Model başarıyla yüklendi")
            
            # Vocabulary yükleme
            char_to_idx_path = self.artifacts_dir / "char_to_idx.json"
            idx_to_char_path = self.artifacts_dir / "idx_to_char.json"
            
            if not char_to_idx_path.exists() or not idx_to_char_path.exists():
                raise FileNotFoundError("Vocabulary dosyaları bulunamadı")
            
            with open(char_to_idx_path, "r", encoding="utf-8") as f:
                self.char_to_idx = json.load(f)
            
            with open(idx_to_char_path, "r", encoding="utf-8") as f:
                idx_to_char_raw = json.load(f)
                self.idx_to_char = {int(k): v for k, v in idx_to_char_raw.items()}
            
            self.vocab_size = len(self.char_to_idx)
            logger.info(f"Vocabulary yüklendi (boyut: {self.vocab_size})")
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            raise
    
    def sample_with_temperature(self, predictions: np.ndarray, temperature: float = 1.0) -> int:
        """
        Temperature-based sampling ile sonraki karakteri seçer
        
        Args:
            predictions: Model çıktısı (probability distribution)
            temperature: Sampling sıcaklığı
                - Düşük (0.2-0.5): Daha deterministik, güvenli çıktı
                - Orta (0.5-1.0): Dengeli
                - Yüksek (1.0+): Daha yaratıcı, riskli
        
        Returns:
            Seçilen karakterin index'i
        """
        predictions = np.asarray(predictions).astype("float64")
        predictions = np.log(predictions + 1e-8) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        
        return np.random.choice(len(predictions), p=predictions)
    
    def generate_text(
        self,
        seed_text: str,
        length: int = 400,
        temperature: float = 0.5
    ) -> str:
        """
        Verilen seed text'ten başlayarak metin üretir
        
        Args:
            seed_text: Başlangıç metni
            length: Üretilecek karakter sayısı
            temperature: Sampling sıcaklığı
        
        Returns:
            Üretilen metin (seed text dahil)
        """
        if self.model is None:
            raise RuntimeError("Model yüklenmemiş!")
        
        generated = seed_text.lower()
        
        for _ in range(length):
            # Son SEQ_LENGTH karakteri al
            seq = generated[-self.seq_length:]
            
            # Padding (seed kısa ise)
            if len(seq) < self.seq_length:
                seq = " " * (self.seq_length - len(seq)) + seq
            
            # Karakterleri index'lere çevir
            x = np.zeros((1, self.seq_length), dtype=np.int32)
            for t, char in enumerate(seq):
                x[0, t] = self.char_to_idx.get(char, 0)
            
            # Tahmin yap
            predictions = self.model.predict(x, verbose=0)[0]
            next_idx = self.sample_with_temperature(predictions, temperature)
            next_char = self.idx_to_char[next_idx]
            
            generated += next_char
        
        logger.info("Metin üretimi tamamlandı")
        return generated


# =========================
# CACHED RESOURCES
# =========================
@st.cache_resource
def load_generator() -> StreamlitTextGenerator:
    """
    Text generator'ı yükler ve cache'ler
    
    Returns:
        Yüklenmiş StreamlitTextGenerator instance
    """
    generator = StreamlitTextGenerator(
        artifacts_dir=Config.ARTIFACTS_DIR,
        seq_length=Config.SEQ_LENGTH
    )
    generator.load_model_and_vocab()
    return generator

# =========================
# STREAMLIT UI
# =========================
def main():
    """Ana Streamlit uygulaması"""
    
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="Tolstoy Style Text Generator",
        page_icon="🎭",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Başlık ve açıklama
    st.title("🎭 Tolstoy Style Text Generator")
    st.markdown(
        """
        Bu uygulama, **LSTM tabanlı character-level** bir derin öğrenme modeli kullanarak 
        **Lev Tolstoy stilinde metin üretimi** yapmaktadır.
        
        Model, *Anna Karenina* ve *War and Peace* eserleri üzerinde eğitilmiştir.
        """
    )
    
    # Model yükleme
    try:
        generator = load_generator()
        
        # Model bilgileri
        with st.expander("ℹ️ Model Bilgileri"):
            st.write(f"**Vocabulary Boyutu:** {generator.vocab_size} karakter")
            st.write(f"**Sequence Length:** {generator.seq_length}")
            st.write(f"**Model Tipi:** Bidirectional LSTM")
            st.write(f"**Artifact Dizini:** `{generator.artifacts_dir}`")
        
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata oluştu: {e}")
        st.info("Lütfen `artifacts/` dizininde gerekli dosyaların olduğundan emin olun.")
        logger.error(f"Model yükleme hatası: {e}")
        return
    
    # Sidebar - Kontrol paneli
    st.sidebar.header("⚙️ Ayarlar")
    
    st.sidebar.markdown("### 📝 Başlangıç Metni")
    seed_text = st.sidebar.text_area(
        "Seed Text",
        value=Config.DEFAULT_SEED,
        height=100,
        help="Model bu metinden devam ederek yeni metin üretecek"
    )
    
    st.sidebar.markdown("### 🌡️ Temperature")
    temperature = st.sidebar.slider(
        "Yaratıcılık Seviyesi",
        min_value=Config.MIN_TEMPERATURE,
        max_value=Config.MAX_TEMPERATURE,
        value=Config.DEFAULT_TEMPERATURE,
        step=0.1,
        help="Düşük: daha tutarlı, Yüksek: daha yaratıcı"
    )
    
    # Temperature açıklaması
    if temperature < 0.5:
        temp_desc = "🔹 **Düşük** - Daha deterministik ve güvenli çıktı"
    elif temperature < 0.9:
        temp_desc = "🔸 **Orta** - Dengeli yaratıcılık"
    else:
        temp_desc = "🔶 **Yüksek** - Daha yaratıcı ve riskli çıktı"
    st.sidebar.caption(temp_desc)
    
    st.sidebar.markdown("### 📏 Üretim Uzunluğu")
    length = st.sidebar.slider(
        "Karakter Sayısı",
        min_value=Config.MIN_LENGTH,
        max_value=Config.MAX_LENGTH,
        value=Config.DEFAULT_LENGTH,
        step=50,
        help="Üretilecek toplam karakter sayısı"
    )
    
    # Üretim butonu
    st.sidebar.markdown("---")
    generate_btn = st.sidebar.button(
        "✍️ Metin Üret",
        type="primary",
        use_container_width=True
    )
    
    # Metin üretimi
    if generate_btn:
        if not seed_text.strip():
            st.warning("⚠️ Lütfen bir seed text girin")
            return
        
        try:
            with st.spinner("🔄 Metin üretiliyor... Lütfen bekleyin."):
                output = generator.generate_text(
                    seed_text=seed_text,
                    length=length,
                    temperature=temperature
                )
            
            # Sonuçları göster
            st.success("✅ Metin başarıyla üretildi!")
            
            # Metin alanı
            st.subheader("📜 Üretilen Metin")
            st.text_area(
                label="",
                value=output,
                height=400,
                label_visibility="collapsed"
            )
            
            # İstatistikler
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Toplam Karakter", len(output))
            with col2:
                st.metric("Kelime Sayısı", len(output.split()))
            with col3:
                st.metric("Satır Sayısı", output.count('\n') + 1)
            
        except Exception as e:
            st.error(f"❌ Metin üretilirken hata oluştu: {e}")
            logger.error(f"Üretim hatası: {e}")
    
    # Alt bilgi
    st.markdown("---")
    st.caption("💡 **Character-level LSTM** | Generative AI Demo | Tolstoy Corpus")
    
    # Sidebar alt bilgi
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center;'>
        <small>
        🎓 <b>Deep Learning Project</b><br>
        Character-Level Text Generation<br>
        LSTM Neural Network
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
