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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image

# 1. TEMİZLİK: Bozuk dosyaları ayıkla
def temizle(dizin):
    for root, dirs, files in os.walk(dizin):
        for file in files:
            dosya_yolu = os.path.join(root, file)
            try:
                with Image.open(dosya_yolu) as img:
                    img.verify()
            except:
                os.remove(dosya_yolu)
                print(f"Silindi: {dosya_yolu}")

temizle('data')

# 1. Veri Artırma ve Hazırlama Kurallarını Tanımlayalım
# Sadece Eğitim verisi için artırma yapıyoruz, doğrulama sadece ölçeklenir.

train_datagen = ImageDataGenerator(
    rescale=1./255,            # Piksel değerlerini 0-1 arasına çeker
    rotation_range=40,         # 40 dereceye kadar rastgele döndür
    zoom_range=0.2,            # Yakınlaştırma yap
    horizontal_flip=True,      # Yatayda çevi
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

#MODEL MİMARİSİ (TRANSFER LEARNING)
# Önceden eğitilmiş MobileNetV2'yi yükle
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Temel katmanları dondur
base_model.trainable = False

# Kendi katmanlarımızı ekleyelim
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(6, activation='softmax')(x) # 6 farklı atık türü için

# Nihai modeli oluştur
model = Model(inputs=base_model.input, outputs=output)

# Modeli derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. EĞİTİM
checkpoint = ModelCheckpoint('atik_modeli.keras', monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3)

print("\n--- Eğitim Başlıyor... ---")
model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[early_stop, checkpoint])