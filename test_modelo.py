import tensorflow as tf
import numpy as np
from PIL import Image
import os

def main():
    try:
        model = tf.keras.models.load_model('modelo_food_advanced_final.h5')
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    try:
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Classes disponíveis: {class_names}")
    except Exception as e:
        print(f"Erro ao carregar nomes das classes: {e}")
        return

    while True:
        image_path = input("Digite o caminho da imagem para testar (ou pressione Enter para usar exemplo): ").strip()
        
        if image_path == "":
            example_images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if example_images:
                image_path = example_images[0]
                print(f"Usando imagem de exemplo: {image_path}")
            else:
                print("Nenhuma imagem de exemplo encontrada!")
                continue
        
        if not os.path.exists(image_path):
            print("Arquivo não encontrado!")
            continue
            
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = image.resize((224, 224))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            print(f"\nResultados da predição:")
            print(f"Imagem: {os.path.basename(image_path)}")
            print(f"Predição: {class_names[predicted_class]}")
            print(f"Confiança: {confidence:.4f} ({confidence*100:.2f}%)")
            top_indices = np.argsort(predictions[0])[::-1][:3]
            print(f"\nTop 3 predições:")
            for i, idx in enumerate(top_indices):
                conf = predictions[0][idx]
                name = class_names[idx]
                print(f"{i+1}. {name}: {conf:.4f} ({conf*100:.2f}%)")
                
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            
        if input("\nDeseja testar outra imagem? (s/n): ").lower() != 's':
            break

if __name__ == "__main__":
    main()