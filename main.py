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