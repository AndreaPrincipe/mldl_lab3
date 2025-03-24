import torch
from torch import nn

# Definisce una neural network personalizzata usando layer convoluionali e un fully connected layer (Linear)
class CustomNet(nn.Module): #classe base per le neural network in pytorch
    def __init__(self): #dice al modello quali operazioni fare
        super(CustomNet, self).__init__()  # inizializza la classe padre
        # Define layers of the neural network, servono per estrarre feature dalle immagini
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2) # 224x224 a 112x112, primo layer: input: 3 canali, output: 64 feature maps, kernel: 3x3, padding: 1 (mantiene dim 224x224), stride:1 (nessuna riduzione della dimensione). I nodi sono i pallini del network in ogni layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # 112x112 → 56x56, meglio mettere stride 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2) # 56x56 → 28x28
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2) # 28x28 → 14x14
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)  # 14x14 → 7x7

        #self.pool = nn.MaxPool2d(2, 2)  # Riduce la dimensione a metà, ma se usi stride=2 nelle convoluzioni per ridurre la dimensione, quindi non serve applicare pooling dopo ogni convoluzione.
        # Add more layers...
        #Prende un vettore appiattito di dimensione 256 * 28 * 28 e lo mappa a 512 neuroni.
        self.fc1 = nn.Linear(512*7*7, 200) # # 512 feature maps, ognuna di 7x7 e 200 is the number of classes in TinyImageNet, questo layer è fully connected
        #L'output convoluzionale viene appiattito (flatten) in un vettore di lunghezza 512 × 7 × 7 = 25088. Il fully connected layer classifica l'immagine in 200 classi di Tiny ImageNet.
        #Vedi tabella sotto

    def forward(self, x):
        # Define forward pass
        # Passaggio attraverso i layer convoluzionali con ReLU
        x = self.conv1(x).relu() # B x 64 x 112 x 112 (Altezza × Larghezza × Canali) e B è la batch size
        x = self.conv2(x).relu() # B x 128 x 56 x 56
        x = self.conv3(x).relu() # B x 256 x 28 x 28
        x = self.conv4(x).relu() # B x 256 x 14 x 14
        x = self.conv5(x).relu() # B x 512 x 7 x 7

        # Flatten prima del fully connected
        x = x.view(x.size(0), -1)  # B x (512 * 7 * 7)
        # Passaggio nel fully connected
        x = self.fc1(x)  # B x 200

        return x