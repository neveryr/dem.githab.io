import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Загрузка меток классов
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
classes = requests.get(imagenet_labels_url).json()

# Загрузка предобученной модели ResNet и перевод её в режим инференса
# Создание пустой модели
model = models.resnet18(pretrained=True) 
model.eval()

# Определение преобразований для изображения
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_top3(image):
    image_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(image_tensor) # ваш код здесь для получения предсказаний от модели
    
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top3_probs, top3_indices = torch.topk(probabilities, 3) 
    
    results = []
    for i in range(3):
        # classes содержит метки классов ImageNet, загруженные ранее
        label = classes[top3_indices[i].item()]
        prob = top3_probs[i].item() * 100  # Проценты
        results.append(f"{label}: {prob:.2f}%")
    
    return results

# Пример использования функции
predictions = predict_top3(image.convert("RGB"))

print("Топ-3 предсказаний:")
for pred in predictions:
    print(pred)
