import cv2
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch
import os
import platform
import gc

class LocalDetector:
    def __init__(self):
        # GPU kullanımını kontrol et
        if torch.cuda.is_available():
            self.device = 'cuda'
            torch.cuda.set_device(0)
            print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # CUDA optimizasyonları
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Bellek yönetimi
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
        elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using Apple Silicon GPU")
            # MPS optimizasyonları
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:
            self.device = 'cpu'
            print("Using CPU - No GPU detected")
            
        print(f"PyTorch Device: {self.device}")
        
        # YOLOv8n modelini yükle
        try:
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            # Model optimizasyonları
            if self.device == 'cuda':
                self.model.fuse()  # Katmanları birleştir
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            raise
        
        # Bellek temizliği
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
    def process_frame(self, frame):
        try:
            # Frame boyutunu küçült
            frame = cv2.resize(frame, (640, 480))
            
            # Frame'i GPU'ya taşı (CUDA için)
            if self.device == 'cuda':
                with torch.cuda.amp.autocast():  # Otomatik karışık hassasiyet
                    results = self.model(frame, device=self.device)
            else:
                results = self.model(frame, device=self.device)
            
            # Sonuçları işle
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Koordinatları al ve CPU'ya taşı
                        box_tensor = box.xyxy[0]
                        if self.device in ['cuda', 'mps']:
                            box_tensor = box_tensor.cpu()
                        x1, y1, x2, y2 = box_tensor.numpy().astype(int)
                        
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        # Minimum güven skoru kontrolü - 0.5'e yükseltildi
                        if confidence < 0.5:  # Düşük güvenilirlikli tespitleri atla
                            continue
                        
                        # Tespit edilen nesneyi çerçeve içine al
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Sınıf adı ve güven skorunu ekle
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Eğer tespit edilen nesne bir insan ise
                        if class_name == 'person':
                            try:
                                # Yüz bölgesini kırp
                                face = frame[y1:y2, x1:x2]
                                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:  # Minimum yüz boyutu kontrolü
                                    # DeepFace analizi yap
                                    analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False, silent=True)
                                    if isinstance(analysis, list):
                                        analysis = analysis[0]
                                    
                                    # Yaş ve cinsiyet bilgisini ekle
                                    age = analysis['age']
                                    gender = analysis['gender']
                                    info = f"Age: {age}, Gender: {gender}"
                                    cv2.putText(frame, info, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            except Exception as e:
                                continue  # Yüz analizi hatalarını sessizce atla
                    except Exception as e:
                        continue  # Nesne işleme hatalarını sessizce atla
            
            return frame
            
        except Exception as e:
            print(f"Frame işleme hatası: {str(e)}")
            return frame  # Hata durumunda orijinal frame'i döndür
        finally:
            # Her frame sonrası bellek temizliği
            if self.device == 'cuda':
                torch.cuda.empty_cache()

class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return None 