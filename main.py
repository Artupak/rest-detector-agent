import cv2
from local_detector import LocalDetector

def main():
    print("Program başlatılıyor...")
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("HATA: Kamera açılamadı!")
        return
    
    # Kamera çözünürlüğünü maksimuma ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Gerçek çözünürlüğü al
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kamera çözünürlüğü: {actual_width}x{actual_height}")
    
    # Detector'ı başlat
    detector = LocalDetector()
    
    # Pencere oluştur
    window_name = 'Nesne Tespiti'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Ekran boyutunu al
    screen_width = 1920  # Varsayılan değerler
    screen_height = 1080
    
    try:
        # Windows için ekran boyutunu al
        from win32api import GetSystemMetrics
        screen_width = GetSystemMetrics(0)
        screen_height = GetSystemMetrics(1)
    except:
        pass
    
    # Pencereyi ekran boyutuna ayarla
    cv2.resizeWindow(window_name, screen_width, screen_height)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("Kontroller:")
    print("- Çıkmak için 'q' tuşuna basın")
    print("- Tam ekran için 'f' tuşuna basın")
    print("- Pencere moduna dönmek için 'ESC' tuşuna basın")
    
    is_fullscreen = True
    
    while True:
        # Frame'i oku
        ret, frame = cap.read()
        if not ret:
            print("HATA: Frame okunamadı!")
            break
        
        # Frame'i işle
        processed_frame = detector.process_frame(frame)
        
        # Frame'i ekran boyutuna ölçekle
        processed_frame = cv2.resize(processed_frame, (screen_width, screen_height))
        
        # Sonucu göster
        cv2.imshow(window_name, processed_frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):  # Tam ekran
            is_fullscreen = True
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == 27:  # ESC - Pencere modu
            is_fullscreen = False
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 