import cv2
import numpy as np
from ultralytics import YOLO
import torch
import gdown
import os

class LocalDetector:
    def __init__(self):
        print("LocalDetector başlatılıyor...")
        try:
            # YOLOv8x modelini yükle
            print("YOLO modeli yükleniyor (yolov8x)...")
            self.yolo = YOLO('yolov8x.pt')
            
            # Yaş-Cinsiyet modelini indir ve yükle
            self.load_age_gender_model()
            
            # Model bilgilerini yazdır
            print(f"Model yüklendi. Mevcut sınıflar: {self.yolo.names}")
            print(f"Model cihazı: {self.yolo.device}")
            
            # GPU kullanımını zorla
            if torch.cuda.is_available():
                self.yolo.to('cuda')
                print(f"GPU aktif: {torch.cuda.get_device_name()}")
            else:
                print("GPU bulunamadı, CPU kullanılacak")
            
            print("Model başarıyla yüklendi!")
            
            # YOLO sınıfları için Türkçe isimler
            self.class_names_tr = {
                'person': 'Insan',
                'bicycle': 'Bisiklet',
                'car': 'Araba',
                'motorcycle': 'Motosiklet',
                'airplane': 'Ucak',
                'bus': 'Otobus',
                'train': 'Tren',
                'truck': 'Kamyon',
                'boat': 'Tekne',
                'cell phone': 'Telefon',
                'book': 'Kitap',
                'laptop': 'Laptop',
                'mouse': 'Mouse',
                'keyboard': 'Klavye',
                'remote': 'Kumanda',
                'cup': 'Bardak',
                'bottle': 'Sise',
                'chair': 'Sandalye',
                'dining table': 'Masa',
                'tv': 'TV',
                'clock': 'Saat',
                'tie': 'Kravat',
                'backpack': 'Sırt Çantası',
                'handbag': 'El Çantası',
                'suitcase': 'Bavul',
                'umbrella': 'Şemsiye',
                'eye glasses': 'Gözlük',
                'watch': 'Kol Saati'
            }
            
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            raise

    def load_age_gender_model(self):
        """Yaş ve cinsiyet tespit modellerini yükle"""
        try:
            # Model dosyalarının URL'leri
            age_model_url = 'https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW'
            gender_model_url = 'https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ'
            age_model_path = 'age_net.caffemodel'
            gender_model_path = 'gender_net.caffemodel'
            
            # Modelleri indir
            if not os.path.exists(age_model_path):
                print("Yaş modeli indiriliyor...")
                gdown.download(age_model_url, age_model_path, quiet=False)
            if not os.path.exists(gender_model_path):
                print("Cinsiyet modeli indiriliyor...")
                gdown.download(gender_model_url, gender_model_path, quiet=False)
            
            # Prototxt dosyaları
            age_proto = "age_deploy.prototxt"
            gender_proto = "gender_deploy.prototxt"
            
            # Prototxt içeriğini oluştur
            age_proto_content = """
            name: "Age Net"
            input: "data"
            input_dim: 1
            input_dim: 3
            input_dim: 227
            input_dim: 227
            
            layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1"
              convolution_param { num_output: 96 kernel_size: 7 stride: 4
                weight_filler { type: "gaussian" std: 0.01 }
                bias_filler { type: "constant" value: 0 } } }
            """
            
            gender_proto_content = """
            name: "Gender Net"
            input: "data"
            input_dim: 1
            input_dim: 3
            input_dim: 227
            input_dim: 227
            
            layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1"
              convolution_param { num_output: 96 kernel_size: 7 stride: 4
                weight_filler { type: "gaussian" std: 0.01 }
                bias_filler { type: "constant" value: 0 } } }
            """
            
            # Prototxt dosyalarını oluştur
            with open(age_proto, 'w') as f:
                f.write(age_proto_content)
            with open(gender_proto, 'w') as f:
                f.write(gender_proto_content)
            
            # Modelleri yükle
            print("Yaş ve cinsiyet modelleri yükleniyor...")
            self.age_net = cv2.dnn.readNet(age_model_path, age_proto)
            self.gender_net = cv2.dnn.readNet(gender_model_path, gender_proto)
            
            # GPU kullanımını ayarla
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
            self.gender_list = ['Erkek', 'Kadın']
            
            print("Yaş ve cinsiyet modelleri başarıyla yüklendi!")
            
        except Exception as e:
            print(f"Yaş-cinsiyet modeli yükleme hatası: {str(e)}")
            raise

    def detect_age_gender(self, face_img):
        """Yüz görüntüsünden yaş ve cinsiyet tespiti yap"""
        try:
            # Görüntüyü yeniden boyutlandır
            face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                            (78.4263377603, 87.7689143744, 114.895847746), 
                                            swapRB=False)
            
            # Cinsiyet tahmini
            self.gender_net.setInput(face_blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Yaş tahmini
            self.age_net.setInput(face_blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            
            return f"{gender}, {age} yaş"
            
        except Exception as e:
            print(f"Yaş-cinsiyet tespiti hatası: {str(e)}")
            return "Belirsiz"

    def process_frame(self, frame):
        try:
            if frame is None:
                print("HATA: Frame boş!")
                return frame
            
            # Kopya frame oluştur
            output_frame = frame.copy()
            h, w, _ = frame.shape
            
            # YOLO ile nesne tespiti
            results = self.yolo(
                frame,
                verbose=False,
                device=0 if torch.cuda.is_available() else 'cpu',
                conf=0.25,  # Güven skorunu 0.20'den 0.25'e yükselttik
                iou=0.40,   
                agnostic_nms=True,
                max_det=60, 
                classes=None 
            )
            
            # Debug için tespit sayısını yazdır
            for result in results:
                boxes = result.boxes
                print(f"Tespit edilen toplam nesne sayısı: {len(boxes)}")
                for box in boxes:
                    cls_name = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    print(f"Tespit: {cls_name}, Güven: {conf:.2f}")
            
            # Tespit edilen nesneleri işle
            detected_objects = {}
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names[cls_id]
                    
                    if cls_name in self.class_names_tr:
                        tr_name = self.class_names_tr[cls_name]
                        detected_objects[tr_name] = detected_objects.get(tr_name, 0) + 1
                        
                        # Nesne kutusunu çiz
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Eğer tespit edilen nesne bir insan ise, yaş ve cinsiyet tespiti yap
                        extra_info = ""
                        if cls_name == 'person':
                            # Yüz bölgesini kırp (başın üst kısmını da içerecek şekilde)
                            face_y1 = max(0, y1 - int((y2-y1)*0.2))  # Başın üst kısmını da al
                            face_img = frame[face_y1:y2, x1:x2]
                            if face_img.size > 0:  # Görüntü geçerli mi kontrol et
                                extra_info = self.detect_age_gender(face_img)
                        
                        # Farklı nesneler için farklı renkler
                        color = (0, 255, 0)  # Varsayılan yeşil
                        if cls_name == 'person':
                            color = (0, 165, 255)  # Turuncu
                        elif cls_name in ['cell phone', 'laptop', 'mouse', 'keyboard']:
                            color = (255, 0, 0)  # Mavi
                        elif cls_name in ['eye glasses', 'watch']:
                            color = (0, 0, 255)  # Kırmızı
                        
                        # Kutuyu daha kalın çiz
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Etiket için daha büyük ve belirgin arka plan
                        label = f"{tr_name}: {conf:.2f}"
                        if extra_info:  # Yaş ve cinsiyet bilgisi varsa ekle
                            label += f" ({extra_info})"
                        
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(output_frame, (x1, y1-label_h-15), (x1+label_w+10, y1), color, -1)
                        
                        # Etiketi daha büyük yaz
                        cv2.putText(output_frame, label, (x1+5, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Tespit edilen nesneleri sağ üst köşeye daha büyük yaz
            if detected_objects:
                text_y = 40
                for obj, count in detected_objects.items():
                    text = f"{obj}: {count}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    x = w - text_w - 20
                    
                    # Yazı için yarı saydam arka plan
                    alpha = 0.7  # Daha opak arka plan
                    overlay = output_frame.copy()
                    cv2.rectangle(overlay, (x-10, text_y-30), (x+text_w+10, text_y+10), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
                    
                    # Daha büyük yazı
                    cv2.putText(output_frame, text, (x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    text_y += 40
            
            return output_frame
            
        except Exception as e:
            print(f"Frame işleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame 