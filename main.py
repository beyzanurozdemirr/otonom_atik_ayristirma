"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Veri yolunu tanımlayalım
base_dir = 'data' # Klasör ismin neyse o

# Resimleri normalize etme ve eğitim/test olarak ayırma
datagen = ImageDataGenerator(
    rescale=1./255,           # Piksel değerlerini 0-1 arasına çeker
    validation_split=0.2      # Verinin %20'sini test için ayırır
)

# Eğitim verilerini yükleyelim
train_data = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),   # Resim boyutlarını standartlaştırır
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Doğrulama (Validation) verilerini yükleyelim
val_data = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
"""

import os

# Veri yolunu kontrol et
data_yolu = 'data'
siniflar = os.listdir(data_yolu)

print(f"Tespit edilen sınıflar: {siniflar}")

for sinif in siniflar:
    yol = os.path.join(data_yolu, sinif)
    # Sadece klasör olanları say (gizli dosyaları atla)
    if os.path.isdir(yol):
        resim_sayisi = len(os.listdir(yol))
        print(f"- {sinif} klasöründe {resim_sayisi} adet resim var.")