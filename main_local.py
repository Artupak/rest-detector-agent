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
    parser.add_argument('--resolution', type=str, default='720p', help='Video resolution (480p, 720p, 1080p)')
    parser.add_argument('--save', action='store_true', help='Save video output')
    args = parser.parse_args()

    # Kamera cihazlarını kontrol et
    if platform.system() == 'Linux':
        try:
            video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
            if not video_devices:
                print("HATA: Hiçbir kamera cihazı bulunamadı (/dev/video* yok)")
                print("Çözüm önerileri:")
                print("1. Kameranın bağlı olduğundan emin olun")
                print("2. Kullanıcınızın video grubuna ekli olduğunu kontrol edin:")
                print("   sudo usermod -a -G video $USER")
                print("3. v4l2 sürücülerinin yüklü olduğunu kontrol edin:")
                print("   sudo apt-get install v4l-utils")
                sys.exit(1)
            print(f"Bulunan kamera cihazları: {', '.join(video_devices)}")
        except Exception as e:
            print(f"HATA: Kamera cihazları kontrol edilirken hata oluştu: {str(e)}")

    # Çözünürlük ayarları
    resolutions = {
        '480p': (640, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        'HD': (1920, 1080),  # Full HD
        '2K': (2560, 1440),  # 2K
        '4K': (3840, 2160)   # 4K
    }
    
    # Varsayılan çözünürlüğü 720p yap
    width, height = resolutions.get(args.resolution, resolutions['720p']) # args.resolution kullanılır veya varsayılan 720p

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
        print(f"Kamera {args.camera_id} açılıyor...")
        cap = cv2.VideoCapture(args.camera_id)
        
        if not cap.isOpened():
            if platform.system() == 'Linux':
                print(f"\nKamera açma hatası detayları:")
                print(f"1. Kamera ID: {args.camera_id}")
                print(f"2. OpenCV sürümü: {cv2.__version__}")
                print("3. Kamera erişim izinlerini kontrol edin:")
                print("   ls -l /dev/video*")
                print("\nÇözüm önerileri:")
                print("1. Farklı bir kamera ID'si deneyin (--camera-id 1 veya 2)")
                print("2. v4l2-ctl --list-devices komutunu çalıştırarak mevcut kameraları listeleyin")
                print("3. sudo apt-get install v4l-utils ile video sürücülerini güncelleyin")
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

        # Video kaydı için ayarlar
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, actual_fps, (actual_width, actual_height))

        # Detector sınıfını başlat
        detector = LocalDetector()

        # Ana pencereyi oluştur
        window_name = 'Object Detection'
        if platform.system() == 'Linux':
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, actual_width, actual_height)
            # Linux'ta tam ekran modunu dene
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # Pencereyi öne getir
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

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

                # Video kaydet
                if args.save:
                    out.write(processed_frame)

                # Sonucu göster
                cv2.imshow(window_name, processed_frame)
                
                # Linux'ta pencere yenileme ve bekleme
                if platform.system() == 'Linux':
                    cv2.waitKey(1)  # Linux'ta pencere yenilemesi için gerekli
                    time.sleep(0.001)  # CPU kullanımını azalt

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
        if args.save and 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        # Linux'ta pencere kapanmasını garantile
        if platform.system() == 'Linux':
            for i in range(5):
                cv2.waitKey(1)
        print("\nProgram kapatıldı.")

if __name__ == "__main__":
    main() 
