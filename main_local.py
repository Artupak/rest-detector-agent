import cv2
from local_detector import LocalDetector
import argparse
import torch
import platform
import time
import psutil
import os
import signal
import sys

def check_display():
    """Check if display is available for GUI windows"""
    if platform.system() == 'Linux':
        if os.environ.get('DISPLAY') is None:
            print("HATA: X11 display bulunamadı. GUI penceresi gösterilemiyor.")
            print("Çözüm için:")
            print("1. X11 yüklü olduğundan emin olun")
            print("2. DISPLAY değişkeninin ayarlı olduğunu kontrol edin")
            print("3. SSH üzerinden çalışıyorsanız X11 forwarding aktif olmalı")
            return False
    return True

def signal_handler(sig, frame):
    print("\nProgram kapatılıyor...")
    cv2.destroyAllWindows()
    sys.exit(0)

def get_system_info():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_reserved() / 1024**3  # GB cinsinden
        return f"CPU: {cpu_percent}% | RAM: {memory_percent}% | GPU Memory: {gpu_memory:.1f}GB"
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        return f"CPU: {cpu_percent}% | RAM: {memory_percent}% | Apple GPU Active"
    else:
        return f"CPU: {cpu_percent}% | RAM: {memory_percent}%"

def main():
    # Ctrl+C sinyalini yakala
    signal.signal(signal.SIGINT, signal_handler)
    
    # Display kontrolü
    if not check_display():
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Local Object Detection')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage if available')
    parser.add_argument('--resolution', type=str, default='480p', help='Video resolution (480p, 720p, 1080p)')
    args = parser.parse_args()

    # Çözünürlük ayarları
    resolutions = {
        '480p': (640, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080)
    }
    width, height = resolutions.get(args.resolution, (640, 480))  # Varsayılan 480p

    # GPU kullanılabilirliğini kontrol et
    if args.gpu:
        if torch.cuda.is_available():
            print(f"NVIDIA GPU bulundu: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Kullanılabilir GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
            print("Apple Silicon GPU bulundu")
        else:
            print("GPU bulunamadı, CPU kullanılacak")

    try:
        # Kamera başlat
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            raise Exception(f"Kamera açılamadı! ID: {args.camera_id}")

        # Kamera özelliklerini ayarla
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 15)  # FPS'i 15'e düşür
        
        # Gerçek çözünürlüğü kontrol et
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Kamera Çözünürlüğü: {actual_width}x{actual_height} @ {actual_fps}fps")

        # Detector sınıfını başlat
        detector = LocalDetector()

        # Ana pencereyi oluştur ve boyutlandır
        window_name = 'Object Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, actual_width, actual_height)
        
        if platform.system() == 'Linux':
            # Linux'ta pencere özelliklerini ayarla
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

        print("Program başlatıldı. Çıkmak için 'q' tuşuna basın.")

        # Performans izleme için değişkenler
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        stats_update_time = time.time()
        system_stats = get_system_info()
        
        # Frame işleme için zaman kontrolü
        frame_delay = 1.0 / 15  # 15 FPS için minimum bekleme süresi
        last_frame_time = time.time()

        while True:
            try:
                current_time = time.time()
                # FPS limitini uygula
                if current_time - last_frame_time < frame_delay:
                    continue
                
                last_frame_time = current_time
                
                # Frame oku
                ret, frame = cap.read()
                if not ret:
                    print("Frame okunamadı!")
                    break

                # Frame'i işle
                processed_frame = detector.process_frame(frame)

                # FPS hesapla
                fps_frame_count += 1
                if fps_frame_count >= 15:  # Her 15 frame'de bir FPS güncelle
                    fps = fps_frame_count / (current_time - fps_start_time)
                    fps_start_time = current_time
                    fps_frame_count = 0

                # Her 2 saniyede bir sistem istatistiklerini güncelle
                if current_time - stats_update_time > 2:
                    system_stats = get_system_info()
                    stats_update_time = current_time

                # Performans bilgilerini ekrana yaz
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, system_stats, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Sonucu göster
                cv2.imshow(window_name, processed_frame)
                
                # Linux'ta pencere yenileme
                if platform.system() == 'Linux':
                    cv2.waitKey(1)  # Linux'ta pencere yenilemesi için gerekli

                # 'q' tuşuna basılırsa çık
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Frame işleme hatası: {str(e)}")
                continue

    except Exception as e:
        print(f"Program hatası: {str(e)}")
    
    finally:
        # Temizlik
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        # Linux'ta pencere kapanmasını garantile
        if platform.system() == 'Linux':
            for i in range(5):
                cv2.waitKey(1)
        print("\nProgram kapatıldı.")

if __name__ == "__main__":
    main() 
