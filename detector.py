import os, cv2, numpy as np, tensorflow as tf
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from datetime import datetime
from PIL import Image

# --- CONSTANTES (sem alterações) ---
CONFIDENCE_THRESHOLD = 0.65
MIN_AREA_PERCENTAGE = 3.0
IOU_SUPPRESSION_THRESH = 0.75
IOU_MERGE_THRESH = 0.50
SOLIDITY_THRESHOLD = 0.30
MORPH_CLOSE_KERNEL = (9, 9)

# --- FUNÇÕES AUXILIARES (sem alterações) ---
def create_output_directory() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"detection_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_image_to_folder(image: np.ndarray, filename: str, folder: str) -> str:
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)
    return path

def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    xa1, ya1, wa, ha = box_a
    xb1, yb1, wb, hb = box_b
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb2, yb2 = xb1 + wb, yb1 + hb
    inter_width = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_height = max(0, min(ya2, yb2) - max(ya1, yb1))
    intersection = inter_width * inter_height
    union = wa * ha + wb * hb - intersection
    return intersection / union if union > 0 else 0.0

def merge_overlapping_detections(detections: list[dict]) -> list[dict]:
    merged = []
    for det in detections:
        flag = False
        for exist in merged:
            if exist["name"] == det["name"] and compute_iou(exist["bbox"], det["bbox"]) > IOU_MERGE_THRESH:
                x1 = min(exist["bbox"][0], det["bbox"][0])
                y1 = min(exist["bbox"][1], det["bbox"][1])
                x2 = max(exist["bbox"][0] + exist["bbox"][2], det["bbox"][0] + det["bbox"][2])
                y2 = max(exist["bbox"][1] + exist["bbox"][3], det["bbox"][1] + det["bbox"][3])
                exist["bbox"] = (x1, y1, x2 - x1, y2 - y1)
                exist["confidence"] = max(exist["confidence"], det["confidence"])
                flag = True
                break
        if not flag:
            merged.append(det.copy())
    return merged

# --- CLASSES DE DETECÇÃO (sem alterações) ---
class PlateDetector:
    def detect_plate(self, image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=150, maxRadius=600)
        mask = None
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            c = circles[np.argmax(circles[:, 2])]
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.circle(mask, (c[0], c[1]), c[2], 255, -1)
        if mask is None:
            lower, upper = np.array([0, 0, 180]), np.array([180, 30, 255])
            white = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((20, 20), np.uint8)
            opened = cv2.morphologyEx(cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
            cont, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cont and cv2.contourArea(max(cont, key=cv2.contourArea)) > 50000:
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillPoly(mask, [max(cont, key=cv2.contourArea)], 255)
        if mask is None:
            mask = np.full(image.shape[:2], 255, np.uint8)
        ys, xs = np.where(mask > 0)
        y1, x1 = max(0, ys.min() - 30), max(0, xs.min() - 30)
        y2, x2 = min(image.shape[0], ys.max() + 30), min(image.shape[1], xs.max() + 30)
        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1, y1)

class FoodSegmenter:
    def segment(self, plate: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
        lower, upper = np.array([0, 0, 220]), np.array([180, 25, 255])
        white = cv2.inRange(hsv, lower, upper)
        not_white = cv2.bitwise_not(white)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        lap = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
        t1 = cv2.threshold(lap, 20, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.Canny(gray, 30, 80)
        t2 = cv2.dilate(edges, np.ones((3,3),np.uint8), 1)
        b1 = cv2.GaussianBlur(gray, (5,5), 1)
        b2 = cv2.GaussianBlur(gray, (15,15), 3)
        t3 = cv2.threshold(cv2.absdiff(b1, b2), 10, 255, cv2.THRESH_BINARY)[1]
        color_mask = np.zeros_like(gray)
        ranges = [([10,30,50],[40,255,255]),([0,50,50],[10,255,255]),([170,50,50],[180,255,255]),([8,40,20],[25,255,200]),([40,40,40],[80,255,255]),([0,20,100],[30,80,220])]
        for l,u in ranges:
            color_mask |= cv2.inRange(hsv,np.array(l),np.array(u))
        sample_pixels = plate[cv2.dilate(white, np.ones((10,10),np.uint8),1) > 0]
        diff_mask = np.zeros_like(gray)
        if sample_pixels.size:
            m,s = sample_pixels.mean(0), sample_pixels.std(0)
            diff_mask = (np.abs(plate.astype(float)-m)>(s*1.5)).any(2).astype(np.uint8)*255
        combined = t1|t2|t3|color_mask|diff_mask
        food_mask = cv2.bitwise_and(plate_mask, not_white & combined)
        food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, np.ones(MORPH_CLOSE_KERNEL,np.uint8))
        food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
        food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        return food_mask

def split_mask_with_watershed(mask: np.ndarray, min_region_area: int = 500):
    distance_map = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance_map, min_distance=20, labels=mask)
    markers = np.zeros(distance_map.shape, int)
    for idx,(r,c) in enumerate(coords,1):
        markers[r,c] = idx
    labels = watershed(-distance_map, markers, mask=mask)
    regions = []
    for label in np.unique(labels):
        if label == 0:
            continue
        single = (labels==label).astype(np.uint8)*255
        cont,_ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cont:
            continue
        contour = cont[0]
        area = cv2.contourArea(contour)
        if area < min_region_area:
            continue
        hull = cv2.convexHull(contour)
        if area / cv2.contourArea(hull) < SOLIDITY_THRESHOLD:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        regions.append({"bbox":(x,y,w,h),"mask":single[y:y+h,x:x+w]})
    return regions

class RegionDetector:
    def detect_regions(self, food_mask: np.ndarray, plate_image: np.ndarray):
        h,w = plate_image.shape[:2]
        total = h*w
        cand = split_mask_with_watershed(food_mask)
        valid = []
        for r in cand:
            x,y,w1,h1 = r["bbox"]
            if (w1*h1)/total*100 < MIN_AREA_PERCENTAGE:
                continue
            region_img = plate_image[y:y+h1,x:x+w1]
            valid.append({"bbox":(x,y,w1,h1),"mask":r["mask"],"region":region_img,"percent":(w1*h1)/total*100})
        return valid

class FoodClassifier:
    def __init__(self, model_path: str, names_path: str):
        self.model = tf.keras.models.load_model(model_path)
        with open(names_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f]
    def classify(self, region_img: np.ndarray, folder: str, idx: int):
        resized = cv2.resize(region_img,(224,224), interpolation=cv2.INTER_LANCZOS4)
        path = save_image_to_folder(resized,f"region_{idx}.jpg",folder)
        array = np.expand_dims(np.array(Image.open(path).convert("RGB").resize((224,224)))/255.0,0)
        preds = self.model.predict(array, verbose=0)[0]
        best = np.argmax(preds)
        return self.class_names[best], float(preds[best])

class AdvancedFoodDetector:
    def __init__(self, model_path: str, names_path: str):
        self.plate_detector = PlateDetector()
        self.segmenter = FoodSegmenter()
        self.region_detector = RegionDetector()
        self.classifier = FoodClassifier(model_path, names_path)
        
    def run(self, image_path: str):
        image = cv2.imread(image_path)
        plate_img, plate_mask, (ox,oy) = self.plate_detector.detect_plate(image)
        folder = create_output_directory()
        save_image_to_folder(plate_img,"plate.jpg",folder)
        save_image_to_folder(plate_mask,"plate_mask.jpg",folder)
        
        food_mask = self.segmenter.segment(plate_img, plate_mask)
        save_image_to_folder(food_mask,"food_mask.jpg",folder)
        
        regions = self.region_detector.detect_regions(food_mask, plate_img)
        detections = []
        for idx,r in enumerate(regions):
            name,conf = self.classifier.classify(r["region"],folder,idx)
            if conf < CONFIDENCE_THRESHOLD:
                continue
            r.update({"name":name,"confidence":conf})
            detections.append(r)
            
        ordered = sorted(detections,key=lambda r:r["bbox"][2]*r["bbox"][3],reverse=True)
        final_detections = []
        for det in ordered:
            if all(compute_iou(det["bbox"],ex["bbox"])<=IOU_SUPPRESSION_THRESH for ex in final_detections):
                final_detections.append(det)
                
        final_detections = merge_overlapping_detections(final_detections)
        
        # Desenhar na imagem do prato (para depuração)
        vis_plate = plate_img.copy()
        for det in final_detections:
            x,y,w,h = det["bbox"]
            cv2.rectangle(vis_plate,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(vis_plate,f"{det['name']}({det['confidence']:.2f})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        save_image_to_folder(vis_plate,"detected_plate.jpg",folder)
        
        # Desenhar na imagem completa original
        vis_full = image.copy()
        for det in final_detections:
            x,y,w,h = det["bbox"]
            # Adicionar o offset (ox, oy) para mapear as coordenadas de volta para a imagem original
            cv2.rectangle(vis_full,(x+ox,y+oy),(x+ox+w,y+oy+h),(255,0,0),3)
            cv2.putText(vis_full,f"{det['name']} ({det['confidence']:.2f})",(x+ox,y+oy-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        save_image_to_folder(vis_full,"detected_full.jpg",folder)
        
        # --- INÍCIO DA CORREÇÃO ---
        
        # 1. Converter a imagem final do OpenCV (BGR) para uma imagem PIL (RGB)
        #    O Flask e o navegador esperam imagens no formato RGB.
        final_image_rgb = cv2.cvtColor(vis_full, cv2.COLOR_BGR2RGB)
        final_image_pil = Image.fromarray(final_image_rgb)
        
        # 2. Retornar os dados das detecções E a imagem PIL processada
        #    Esta é a mudança principal para corrigir o erro.
        return final_detections, final_image_pil
        
        # --- FIM DA CORREÇÃO ---

# --- BLOCO DE EXECUÇÃO PRINCIPAL (sem alterações) ---
if __name__ == "__main__":
    import sys, os
    img = sys.argv[1] if len(sys.argv)>1 else None
    if not img or not os.path.exists(img):
        print("Usage: python detector.py <image_path>")
        sys.exit(1)
    # ATENÇÃO: Os nomes dos ficheiros do modelo devem corresponder
    det = AdvancedFoodDetector("modelo_food_advanced_final.h5","class_names.txt")
    # A chamada `run` agora retornará uma tupla, mas para este script de linha de comando, só nos importamos com o primeiro elemento.
    detections_data, _ = det.run(img)
    for d in detections_data:
        x,y,w,h = d["bbox"]
        print(f"{d['name']}({d['confidence']:.2f}) at {x},{y},{w}x{h}")
