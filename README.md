# Real-Time Object Detection Agent

A high-performance, local-first object detection system that leverages YOLOv8 and DeepFace for real-time object detection and face analysis. The system automatically utilizes the best available hardware acceleration for optimal performance.

## Features

- **Real-time Object Detection**: Detects people, vehicles, and bicycles using YOLOv8
- **Face Analysis**: Provides age and gender analysis for detected people using DeepFace
- **Hardware Acceleration**:
  - NVIDIA GPU (CUDA) support for maximum performance
  - Apple Silicon (MPS) optimization for Mac users
  - CPU fallback support
- **Local Processing**: All processing happens locally without cloud dependencies
- **High Performance**: 
  - 13-15ms inference time on Apple Silicon
  - Optimized CUDA performance with mixed precision and cuDNN

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Required packages listed in `requirements.txt`

### Hardware Requirements (one of the following):
- NVIDIA GPU with CUDA support
- Apple Silicon Mac
- Modern CPU (fallback option)
## Usage

Run the main application:
```bash
python main_local.py
```

The system will automatically detect and use the best available hardware acceleration:
1. NVIDIA GPU (if available with CUDA)
2. Apple Silicon GPU (if on compatible Mac)
3. CPU (fallback option)

## Performance

- NVIDIA GPU: Optimized with CUDA, mixed precision, and cuDNN
- Apple Silicon: 13-15ms inference time
- CPU: 50-70ms inference time

---

# Gerçek Zamanlı Nesne Algılama Ajanı

YOLOv8 ve DeepFace kullanarak gerçek zamanlı nesne tespiti ve yüz analizi yapan, yüksek performanslı, yerel öncelikli bir nesne algılama sistemi. Sistem, optimal performans için otomatik olarak mevcut en iyi donanım hızlandırmasını kullanır.

## Özellikler

- **Gerçek Zamanlı Nesne Tespiti**: YOLOv8 kullanarak insan, araç ve bisiklet tespiti
- **Yüz Analizi**: DeepFace kullanarak tespit edilen kişiler için yaş ve cinsiyet analizi
- **Donanım Hızlandırma**:
  - Maksimum performans için NVIDIA GPU (CUDA) desteği
  - Mac kullanıcıları için Apple Silicon (MPS) optimizasyonu
  - CPU yedek desteği
- **Yerel İşleme**: Tüm işlemler bulut bağımlılığı olmadan yerel olarak gerçekleşir
- **Yüksek Performans**: 
  - Apple Silicon'da 13-15ms çıkarım süresi
  - Karma hassasiyet ve cuDNN ile optimize edilmiş CUDA performansı

## Gereksinimler

- Python 3.8+
- PyTorch
- OpenCV
- `requirements.txt` dosyasında listelenen gerekli paketler

### Donanım Gereksinimleri (aşağıdakilerden biri):
- CUDA destekli NVIDIA GPU
- Apple Silicon Mac
- Modern CPU (yedek seçenek)

## Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/kullanıcıadınız/rest-detector-agent.git
cd rest-detector-agent
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

Ana uygulamayı çalıştırın:
```bash
python main_local.py
```

Sistem otomatik olarak mevcut en iyi donanım hızlandırmasını tespit edip kullanacaktır:
1. NVIDIA GPU (CUDA ile kullanılabilir ise)
2. Apple Silicon GPU (uyumlu Mac'te)
3. CPU (yedek seçenek)

## Performans

- NVIDIA GPU: CUDA, karma hassasiyet ve cuDNN ile optimize edilmiş
- Apple Silicon: 13-15ms çıkarım süresi
- CPU: 50-70ms çıkarım süresi 
