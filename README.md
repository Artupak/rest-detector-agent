# Real-Time Object Detection Agent (rest-detector-agent)

High-performance, local-first object detection and face analysis system. This agent is designed to run inference at the edge with automatic hardware acceleration discovery, ensuring maximum privacy and minimal latency without cloud dependencies.

*(Turkce dokumantasyon asagidadir / Turkish documentation is below)*

## Key Features

* Real-Time Detection: Optimized YOLOv8 implementation for tracking people, vehicles, and bicycles.
* Face Analysis: Integrated DeepFace pipeline for instant age and gender estimation of detected individuals.
* Auto-Hardware Acceleration:
  * NVIDIA GPU: Full CUDA and cuDNN optimization with Mixed Precision inference.
  * Apple Silicon: Native MPS (Metal Performance Shaders) support for M1/M2/M3 chips.
  * CPU Fallback: Optimized AVX/OpenMP execution for systems without dedicated GPUs.
* Local-First (Privacy Centric): All biometric and visual data is processed on-device. No external API calls are made.

## Performance Metrics

*Tests conducted on a standard real-time video stream (640x480 resolution).*

Hardware | Backend | Inference Time | Total Pipeline Latency
--- | --- | --- | ---
Apple Silicon (M2) | MPS | ~13-15ms | ~45ms
NVIDIA RTX 3060 | CUDA | ~6-10ms | ~25ms
Intel i7 (12th Gen) | CPU | ~50-70ms | ~120ms

## Installation

1. Clone the Repository:
git clone https://github.com/Artupak/rest-detector-agent.git
cd rest-detector-agent

2. Set Up Environment:
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

3. Install Dependencies:
pip install -r requirements.txt

## Usage

Run the agent with the local detector script:
python main_local.py

The system will automatically initialize the most capable hardware found on the host machine in the following priority: CUDA > MPS > CPU.

## System Architecture

The agent operates through a sequential pipeline:

1. Frame Capture: OpenCV handles the raw video stream I/O.
2. Detection Layer: YOLOv8 (Small/Nano) identifies bounding boxes.
3. Analysis Layer: Detected 'person' objects are cropped and passed to the DeepFace engine.
4. Hardware Dispatcher: A custom abstraction layer maps PyTorch tensors to the optimal device (CUDA/MPS).

## Known Limitations & Lessons Learned (Developer Notes)

This project was built to understand the end-to-end integration of deep learning models in edge environments. As a developer focused on architectural efficiency, I have identified the following bottlenecks in the current version and plan to address them in future iterations:

* [Current] Synchronous Pipeline Bottleneck: YOLO and DeepFace currently operate in a serial manner. On high-traffic streams, synchronous model inference causes frame-dropping and increases total pipeline latency.
  * Next Step: Implement an Asynchronous Multi-threading/Multiprocessing architecture to decouple capture, detection, and analysis.

* [Current] Python GIL Constraints: The Python Global Interpreter Lock (GIL) limits true parallelism during high-load real-time video processing.
  * Next Step: Port the core inference engine to C++ using hardware accelerators like TensorRT (NVIDIA) or OpenVINO (Intel) to bypass Python's I/O and processing bottlenecks.

* [Current] Memory Management: Continuous loading/unloading of face models in VRAM without a watchdog.
  * Next Step: Implement a strict memory management system and persistent VRAM caching to prevent 'Out of Memory' (OOM) errors during continuous 24/7 runs.

---

# Gercek Zamanli Nesne ve Yuz Algilama Ajani (rest-detector-agent)

YOLOv8 ve DeepFace kullanarak gercek zamanli nesne tespiti ve yuz analizi yapan, yuksek performansli, yerel oncelikli (local-first) bir yapay zeka ajani. Sistem, bulut bagimliligi olmadan maksimum gizlilik saglamak ve optimum performans icin cihazdaki en iyi donanim hizlandiricisini otomatik olarak bulmak uzere tasarlanmistir.

## Temel Ozellikler

* Gercek Zamanli Tespit: Insan, arac ve bisiklet takibi icin optimize edilmis YOLOv8 entegrasyonu.
* Yuz Analizi: Tespit edilen kisilerin aninda yas ve cinsiyet tahmini icin DeepFace entegrasyonu.
* Otomatik Donanim Hizlandirma:
  * NVIDIA GPU: Karma hassasiyet (Mixed Precision) cikarimi ile tam CUDA ve cuDNN optimizasyonu.
  * Apple Silicon: M1/M2/M3 cipleri icin yerel MPS (Metal Performance Shaders) destegi.
  * CPU Yedeklemesi: GPU olmayan sistemler icin optimize edilmis AVX/OpenMP calismasi.
* Yerel Calisma (Gizlilik Odakli): Tum biyometrik ve gorsel veriler cihaz uzerinde islenir. Disariya hicbir API istegi yapilmaz.

## Performans Metrikleri

*Testler standart bir gercek zamanli video akisinda (640x480 cozunurluk) gerceklestirilmistir.*

Donanim | Altyapi | Cikarim Suresi (Inference) | Toplam Gecikme (Pipeline)
--- | --- | --- | ---
Apple Silicon (M2) | MPS | ~13-15ms | ~45ms
NVIDIA RTX 3060 | CUDA | ~6-10ms | ~25ms
Intel i7 (12. Nesil) | CPU | ~50-70ms | ~120ms

## Kurulum

1. Repoyu Klonlayin:
git clone https://github.com/Artupak/rest-detector-agent.git
cd rest-detector-agent

2. Ortami Hazirlayin:
python -m venv venv
source venv/bin/activate  # Linux/macOS
# veya
.\venv\Scripts\activate  # Windows

3. Bagimliliklari Yukleyin:
pip install -r requirements.txt

## Kullanim

Ajani yerel tespit script'i ile calistirin:
python main_local.py

Sistem, makinede bulunan en yetenekli donanimi su oncelik sirasina gore otomatik olarak baslatacaktir: CUDA > MPS > CPU.

## Sistem Mimarisi

Ajan ardisik bir islem hatti (pipeline) uzerinden calisir:

1. Kare Yakalama (Frame Capture): OpenCV ham video akisini (I/O) yonetir.
2. Tespit Katmani (Detection Layer): YOLOv8 nesne sinir kutularini (bounding boxes) belirler.
3. Analiz Katmani (Analysis Layer): Tespit edilen 'insan' nesneleri kirpilir ve DeepFace motoruna iletilir.
4. Donanim Yonlendiricisi (Hardware Dispatcher): Ozel bir soyutlama katmani, PyTorch tensorlerini en uygun cihaza (CUDA/MPS) yonlendirir.

## Bilinen Kisitlamalar ve Gelecek Mimarisi (Gelistirici Notlari)

Bu proje, derin ogrenme (deep learning) modellerinin uc cihazlarda (edge) uctan uca entegrasyonunu kavramak icin tasarlanmistir. Mimari verimlilige odaklanan bir gelistirici olarak, mevcut surumdeki darboğazlarin farkindayim ve bir sonraki asamada bunlari cozmeyi hedefliyorum:

* [Mevcut] Senkron Islem Darboğazi: YOLO ve DeepFace su anda ardisik (serial) bir sekilde calismaktadir. Gercek zamanli video isleme senaryolarinda, bu senkron model cikarimi (inference) darboğazlari yaratmakta ve genel performansi etkileyerek kare atlamalarina (frame-drop) neden olabilmektedir.
  * Sonraki Adim: Goruntu yakalama, tespit ve analiz sureclerini birbirinden ayirmak (decouple) icin Asenkron Islem Yonetimi (Multithreading/Multiprocessing) mimarisine gecis.

* [Mevcut] Python GIL Kisitlamalari: Python'un Global Interpreter Lock (GIL) mekanizmasi, yuksek yuk altindaki gercek zamanli video islemlerinde gercek paralelligi engellemektedir.
  * Sonraki Adim: Python'un I/O darboğazlarini asmak ve performansi optimize etmek icin ana cikarim motorunu TensorRT (NVIDIA) veya OpenVINO (Intel) gibi donanim hizlandiricilari kullanarak C++ ortamina tasmak.

* [Mevcut] Bellek Yonetimi: Modellerin VRAM icerisinde surekli yuklenip bosaltilmasi, bir watchdog mekanizmasi olmadan uzun sureli calismalarda bellek sizintisina yol acabilir.
  * Sonraki Adim: 7/24 kesintisiz calismalarda 'Out of Memory' (OOM) hatalarini onlemek icin kati bir bellek yonetim sistemi ve kalici VRAM onbelleklemesi (caching) uygulamak.
