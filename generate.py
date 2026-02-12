"""
Text Generation Module for Character-Level LSTM Model
======================================================
Bu modül, eğitilmiş LSTM modelini kullanarak Tolstoy stilinde metin üretir.

Temel işlevler:
- Model ve vocabulary yükleme
- Temperature-based sampling ile metin üretimi
- CLI arayüzü ile kolay kullanım
- Farklı seed text'ler ve parametrelerle deneme yapma
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

# TensorFlow uyarılarını kapat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN uyarılarını kapat

import numpy as np
from tensorflow.keras.models import load_model, Model

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextGenerator:
    """LSTM modeli ile metin üretme sınıfı"""
    
    def __init__(self, artifacts_dir: str = "artifacts", seq_length: int = 40):
        """
        Args:
            artifacts_dir: Model ve vocabulary dosyalarının bulunduğu dizin
            seq_length: Girdi sequence uzunluğu (model eğitimindekiyle aynı olmalı)
        """
        self.script_dir = Path(__file__).parent
        self.artifacts_dir = (self.script_dir / artifacts_dir).resolve()
        self.seq_length = seq_length
        
        self.model: Model = None
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[str, str] = {}
        self.vocab_size: int = 0
        
        logger.info(f"Artifacts directory: {self.artifacts_dir}")
        
    def load_model_and_vocab(self, verbose: bool = True) -> None:
        """Model ve vocabulary dosyalarını yükler"""
        try:
            # Model yükleme
            model_path = self.artifacts_dir / "best_model.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            
            if verbose:
                print("🔄 Model yükleniyor...")
            self.model = load_model(model_path)
            if verbose:
                print("✅ Model başarıyla yüklendi")
            
            # Vocabulary yükleme
            char_to_idx_path = self.artifacts_dir / "char_to_idx.json"
            idx_to_char_path = self.artifacts_dir / "idx_to_char.json"
            
            if not char_to_idx_path.exists() or not idx_to_char_path.exists():
                raise FileNotFoundError("Vocabulary dosyaları bulunamadı")
            
            with open(char_to_idx_path, "r", encoding="utf-8") as f:
                self.char_to_idx = json.load(f)
            
            with open(idx_to_char_path, "r", encoding="utf-8") as f:
                self.idx_to_char = json.load(f)
            
            self.vocab_size = len(self.char_to_idx)
            if verbose:
                print(f"✅ Vocabulary yüklendi (boyut: {self.vocab_size})")
            
        except Exception as e:
            logger.error(f"Model veya vocabulary yüklenirken hata oluştu: {e}")
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
        temperature: float = 0.5,
        verbose: bool = False
    ) -> str:
        """
        Verilen seed text'ten başlayarak metin üretir
        
        Args:
            seed_text: Başlangıç metni
            length: Üretilecek karakter sayısı
            temperature: Sampling sıcaklığı
            verbose: İlerleme mesajlarını göster
        
        Returns:
            Üretilen metin (seed text dahil)
        """
        if self.model is None:
            raise RuntimeError("Model yüklenmemiş. Önce load_model_and_vocab() çağırın.")
        
        generated = seed_text.lower()
        
        if verbose:
            print(f"⏳ Metin üretiliyor (temperature: {temperature})...")
        
        for i in range(length):
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
            next_char = self.idx_to_char[str(next_idx)]
            
            generated += next_char
            
            if verbose:
                logger.debug(f"{i + 1}/{length} karakter üretildi")
        
        logger.info("Metin üretimi tamamlandı")
        return generated
    
    def generate_multiple(
        self,
        seed_text: str,
        length: int = 400,
        temperatures: Tuple[float, ...] = (0.2, 0.5, 1.0),
        verbose: bool = False
    ) -> Dict[float, str]:
        """
        Farklı temperature değerleriyle birden fazla metin üretir
        
        Args:
            seed_text: Başlangıç metni
            length: Üretilecek karakter sayısı
            temperatures: Denenecek temperature değerleri
            verbose: İlerleme mesajlarını göster
        
        Returns:
            Temperature -> üretilen metin dictionary'si
        """
        results = {}
        
        print(f"\n📝 Seed Text: '{seed_text}'")
        print(f"📊 Karakter sayısı: {length}")
        print(f"🌡️  Temperature değerleri: {temperatures}\n")
        
        for i, temp in enumerate(temperatures, 1):
            print(f"\n{'=' * 70}")
            print(f"  [{i}/{len(temperatures)}] Temperature: {temp}")
            print('=' * 70)
            
            generated_text = self.generate_text(seed_text, length, temp, verbose)
            results[temp] = generated_text
            print(generated_text)
            print()
        return results


def main():
    """Ana fonksiyon - CLI arayüzü"""
    parser = argparse.ArgumentParser(
        description="LSTM modeli ile Tolstoy stilinde metin üretimi"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="the old man looked at",
        help="Başlangıç metni (seed text)"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=400,
        help="Üretilecek karakter sayısı"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 1.0],
        help="Temperature değerleri (örn: --temperature 0.5 veya --temperature 0.2 0.5 1.0)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Model ve vocabulary dosyalarının bulunduğu dizin"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=40,
        help="Girdi sequence uzunluğu (model eğitimindekiyle aynı olmalı)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detaylı ilerleme mesajları göster"
    )
    
    args = parser.parse_args()
    
    try:
        print("\n" + "=" * 70)
        print("  🎭 Tolstoy Style Text Generator")
        print("=" * 70)
        
        # Generator oluştur ve model yükle
        generator = TextGenerator(
            artifacts_dir=args.artifacts_dir,
            seq_length=args.seq_length
        )
        generator.load_model_and_vocab(verbose=True)
        
        # Metin üret
        generator.generate_multiple(
            seed_text=args.seed,
            length=args.length,
            temperatures=tuple(args.temperature),
            verbose=args.verbose
        )
        
        print("=" * 70)
        print("✅ Metin üretimi tamamlandı!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Hata oluştu: {e}")
        raise


if __name__ == "__main__":
    main()
