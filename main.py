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
#VERİ DOĞRULAMA
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
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Veri Artırma ve Hazırlama Kurallarını Tanımlayalım
# Sadece Eğitim verisi için artırma yapıyoruz, doğrulama sadece ölçeklenir.
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Piksel değerlerini 0-1 arasına çeker
    rotation_range=40,         # 40 dereceye kadar rastgele döndür
    width_shift_range=0.2,     # Yatayda %20 kaydır
    height_shift_range=0.2,    # Dikeyde %20 kaydır
    shear_range=0.2,           # Eğriltme uygula
    zoom_range=0.2,            # Yakınlaştırma yap
    horizontal_flip=True,      # Yatayda çevi
    fill_mode='nearest',       # Kaydırma sonrası boşlukları en yakın piksellerle doldur
    validation_split=0.2       # Verinin %20'sini test/doğrulama için ayır
)

# 2. Eğitim Verilerini Yükleyelim
train_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(224, 224),    # Tüm resimleri 224x224 yapar
    batch_size=32,             # Her seferinde 32 resim işler
    class_mode='categorical',  # 6 farklı sınıfımız olduğu için
    subset='training'          # %80'lik eğitim kısmını al
)

# 3. Doğrulama (Validation) Verilerini Yükleyelim
validation_generator = train_datagen.flow_from_directory(
    'data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'        # %20'lik test kısmını al
)

print("Sınıf İndeksleri: ", train_generator.class_indices)