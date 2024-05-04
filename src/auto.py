import clip
import torch
from PIL import Image
import os
from torch import nn

# Cargar el modelo CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Función para cargar y procesar una imagen
def process_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    return image

# Carpeta con tus imágenes de moda
image_folder = './data/images/'
image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # b, 16, 112, 112
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # b, 32, 56, 56
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 28, 28
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 14, 14
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # b, 256, 7, 7
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # b, 128, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # b, 64, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # b, 32, 56, 56
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # b, 16, 112, 112
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 224, 224
            nn.Sigmoid()  # Output as an image with pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from torchvision import datasets, transforms

# Define transformaciones comunes, como cambiar el tamaño de las imágenes y convertirlas en tensores
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes para que coincidan con las dimensiones de entrada del modelo
    transforms.ToTensor(),          # Convertir imágenes en tensores de PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización como se requiere para modelos preentrenados
])

# Crear un Dataset
dataset = datasets.ImageFolder(root='./data/images/', transform=transform)

# Supongamos que 'data_loader' es tu DataLoader para las imágenes
num_epochs = 10
for epoch in range(num_epochs):
    for data in data_loader:
        img = data.to(device)
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def get_embedding(image):
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(image.to(device))
        return embedding

# Diccionario para almacenar los embeddings
embeddings = {}

# Generar embeddings para cada imagen
for image_file in image_files:
    image = process_image(image_file)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    embeddings[image_file] = image_features

print("Embeddings generated for all images.")

# Guardar los embeddings en un archivo
torch.save(embeddings, './data/image_embeddings.pt')