"""
Text Preprocessing Module for Character-Level Generative Model
================================================================
Bu modül metin verilerini karakter seviyesinde işler ve model eğitimi için hazırlar.

Temel işlevler:
- Metin dosyalarını okuma ve birleştirme
- Vocabulary oluşturma (karakter sözlüğü)
- Sequence oluşturma (eğitim için giriş-çıkış çiftleri)
- Preprocessed veriyi kaydetme
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import pickle

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Metin ön işleme sınıfı"""
    
    def __init__(self, data_dir: str, artifacts_dir: str):
        """
        Args:
            data_dir: Metin dosyalarının bulunduğu dizin
            artifacts_dir: İşlenmiş verilerin kaydedileceği dizin
        """
        # Script'in bulunduğu dizine göre mutlak path oluştur
        self.script_dir = Path(__file__).parent
        self.data_dir = (self.script_dir / data_dir).resolve()
        self.artifacts_dir = (self.script_dir / artifacts_dir).resolve()
        
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Artifacts directory: {self.artifacts_dir}")
    
    def load_texts(self) -> str:
        """
        Data dizinindeki tüm .txt dosyalarını okur ve birleştirir
        
        Returns:
            Birleştirilmiş ve temizlenmiş metin
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data dizini bulunamadı: {self.data_dir}")
        
        texts = []
        txt_files = list(self.data_dir.glob("*.txt"))
        
        if not txt_files:
            raise FileNotFoundError(f"Data dizininde .txt dosyası bulunamadı: {self.data_dir}")
        
        logger.info(f"{len(txt_files)} metin dosyası bulundu")
        
        for file_path in txt_files:
            logger.info(f"Okunuyor: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().lower()
                texts.append(content)
                logger.info(f"{file_path.name}: {len(content)} karakter")
        
        combined_text = "\n".join(texts)
        logger.info(f"Toplam metin uzunluğu: {len(combined_text)} karakter")
        
        return combined_text
    
    def clean_text(self, text: str) -> str:
        """
        Metni temizler ve normalize eder
        
        Args:
            text: Ham metin
            
        Returns:
            Temizlenmiş metin
        """
        # Sadece harfler, rakamlar ve temel noktalama işaretleri
        cleaned_text = re.sub(r"[^a-z0-9\s.,;:!?'\n]", "", text)
        logger.info(f"Temizleme sonrası metin uzunluğu: {len(cleaned_text)} karakter")
        return cleaned_text
    
    def build_vocabulary(self, text: str) -> None:
        """
        Metinden vocabulary oluşturur
        
        Args:
            text: İşlenecek metin
        """
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        logger.info(f"Vocabulary boyutu: {self.vocab_size}")
        logger.info(f"Karakterler: {chars}")
    
    def create_sequences(
        self, 
        text: str, 
        sequence_length: int = 40, 
        step: int = 3
    ) -> Tuple[List[str], List[str]]:
        """
        Metinden eğitim sequence'leri oluşturur
        
        Args:
            text: İşlenecek metin
            sequence_length: Her sequence'in uzunluğu
            step: Sliding window adım boyutu
            
        Returns:
            (input_sequences, target_chars) tuple'ı
        """
        sentences = []
        next_chars = []
        
        for i in range(0, len(text) - sequence_length, step):
            sentences.append(text[i:i + sequence_length])
            next_chars.append(text[i + sequence_length])
        
        logger.info(f"Oluşturulan sequence sayısı: {len(sentences)}")
        logger.info(f"Sequence uzunluğu: {sequence_length}")
        logger.info(f"Step boyutu: {step}")
        
        return sentences, next_chars
    
    def save_artifacts(
        self, 
        sentences: List[str], 
        next_chars: List[str]
    ) -> None:
        """
        İşlenmiş verileri ve vocabulary'yi kaydeder
        
        Args:
            sentences: Input sequence'leri
            next_chars: Target karakterler
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Vocabulary mapping'lerini kaydet
        char_to_idx_path = self.artifacts_dir / "char_to_idx.json"
        idx_to_char_path = self.artifacts_dir / "idx_to_char.json"
        
        with open(char_to_idx_path, "w", encoding="utf-8") as f:
            json.dump(self.char_to_idx, f, ensure_ascii=False, indent=2)
        logger.info(f"Kaydedildi: {char_to_idx_path}")
        
        # idx_to_char için integer key'leri string'e çevir (JSON için)
        idx_to_char_str = {str(k): v for k, v in self.idx_to_char.items()}
        with open(idx_to_char_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_char_str, f, ensure_ascii=False, indent=2)
        logger.info(f"Kaydedildi: {idx_to_char_path}")
        
        # Preprocessed sequence'leri kaydet
        sequences_path = self.artifacts_dir / "sequences.pkl"
        with open(sequences_path, "wb") as f:
            pickle.dump({
                'sentences': sentences,
                'next_chars': next_chars,
                'vocab_size': self.vocab_size
            }, f)
        logger.info(f"Kaydedildi: {sequences_path}")
        
        # Özet bilgileri kaydet
        summary_path = self.artifacts_dir / "preprocessing_summary.json"
        summary = {
            'vocab_size': self.vocab_size,
            'num_sequences': len(sentences),
            'sequence_length': len(sentences[0]) if sentences else 0,
            'total_chars': sum(len(s) for s in sentences) + len(next_chars)
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Kaydedildi: {summary_path}")
    
    def process(
        self, 
        sequence_length: int = 40, 
        step: int = 3
    ) -> None:
        """
        Tam preprocessing pipeline'ını çalıştırır
        
        Args:
            sequence_length: Sequence uzunluğu
            step: Sliding window step boyutu
        """
        logger.info("=" * 50)
        logger.info("Text Preprocessing Başlatılıyor")
        logger.info("=" * 50)
        
        # 1. Metinleri yükle
        text = self.load_texts()
        
        # 2. Metni temizle
        text = self.clean_text(text)
        
        # 3. Vocabulary oluştur
        self.build_vocabulary(text)
        
        # 4. Sequence'ler oluştur
        sentences, next_chars = self.create_sequences(text, sequence_length, step)
        
        # 5. Artifacts'ı kaydet
        self.save_artifacts(sentences, next_chars)
        
        logger.info("=" * 50)
        logger.info("Preprocessing Tamamlandı!")
        logger.info("=" * 50)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='Metin verilerini character-level model için ön işler'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Metin dosyalarının bulunduğu dizin (default: ../data)'
    )
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='../artifacts',
        help='İşlenmiş verilerin kaydedileceği dizin (default: ../artifacts)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=40,
        help='Her sequence\'in karakter uzunluğu (default: 40)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=3,
        help='Sliding window adım boyutu (default: 3)'
    )
    
    args = parser.parse_args()
    
    try:
        preprocessor = TextPreprocessor(args.data_dir, args.artifacts_dir)
        preprocessor.process(args.sequence_length, args.step)
    except Exception as e:
        logger.error(f"Hata oluştu: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
