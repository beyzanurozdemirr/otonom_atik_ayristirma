import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import gradio as gr

# 1. Modeli Yükle
model_yolu = 'atik_modeli.keras'
if os.path.exists(model_yolu):
    model = tf.keras.models.load_model(model_yolu)
else:
    print("HATA: model dosyası bulunamadı!")
    model = None

siniflar = ['Karton', 'Cam', 'Metal', 'Kağıt', 'Plastik', 'Çöp']

# --- ÖZEL MAVİ TEMA CSS ---
custom_css = """
body { background: linear-gradient(135deg, #021d33, #0b3954, #087e8b); color: white; font-family: 'Segoe UI', sans-serif; }
.gradio-container { border-radius: 15px; border: 2px solid #00d2ff; background-color: rgba(0,0,0,0.5) !important; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
#title { text-align: center; color: #00d2ff; text-shadow: 2px 2px 5px #000; }
button.primary { background: linear-gradient(45deg, #00d2ff, #3a7bd5) !important; border: none !important; font-weight: bold !important; transition: 0.3s !important; }
button.primary:hover { transform: scale(1.02); box-shadow: 0 0 20px #00d2ff; }
"""


# 2. Analiz Fonksiyonu
def analiz_et(img):
    if img is None: return None, "Görüntü algılanamadı."

    # Ön İşleme
    img_resized = tf.image.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.array(img_array, copy=True) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    preds = model.predict(img_array)[0]
    results = {siniflar[i]: float(preds[i]) for i in range(6)}

    # Tavsiye Sistemi
    top_label = siniflar[np.argmax(preds)]
    if top_label == 'Çöp':
        msg = "⚠️ GERİ DÖNÜŞTÜRÜLEMEZ: Lütfen evsel atık kutusuna atın."
    else:
        msg = f"✅ GERİ DÖNÜŞTÜRÜLEBİLİR: Bu bir {top_label} atığıdır."

    return results, msg


# 3. Blocks ile Arayüz Tasarımı
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# ♻️ AI Atık Ayrıştırma Paneli", elem_id="title")
    gr.Markdown("Balıkesir Üniversitesi - Bilgisayar Mühendisliği Projesi")

    with gr.Row():
        with gr.Column():
            in_img = gr.Image(label="Atık Resmini Buraya Bırak")
            btn = gr.Button("🔍 SİSTEMİ ÇALIŞTIR", variant="primary")

        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Analiz Sonuçları")
            out_text = gr.Textbox(label="Yapay Zeka Kararı", interactive=False)

    btn.click(fn=analiz_et, inputs=in_img, outputs=[out_label, out_text])

if __name__ == "__main__":
    demo.launch(share=False)