import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import cv2 as cv
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import neuronix

neuronix.set_seeds(50, 50)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image and label from the original dataset
        image, label = self.dataset[idx]

        # Unsqueeze the label
        label = torch.tensor(label, dtype=torch.long)  # Change dtype to long
        return image, label

class CNN_Model(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
            )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*32*32, 
                      out_features=output_shape)
        )

    def forward(self, x:torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

dataset = ImageFolder(root='datasets/', transform=transform)
class_names = dataset.classes
dataset= CustomDataset(dataset=dataset)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

model_0 = CNN_Model(input_shape=3, hidden_units=60, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()  # Change to CrossEntropyLoss
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

start_time = timer()

train_test_results = neuronix.train_and_eval(model=model_0,
                        train_loader=train_dataloader,
                        test_loader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        accuracy_fn=neuronix.accuracy_fn,
                        epochs=15,
                        device=device
)

stop_time = timer()

neuronix.print_train_time(start=start_time, end=stop_time, device=device)

neuronix.plot_loss_curves(train_test_results)
neuronix.plot_confusion_matrix(train_test_results['test_labels'], train_test_results['test_predictions'], class_names=class_names)

image = cv.imread(r"test_dataset\testing-images\(20).jpg", cv.IMREAD_COLOR)
image = cv.resize(image, (128, 128))

# Convert the image to a PIL Image and apply the same transform
image_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
data = transform(image_pil)
data = data.unsqueeze(0)  # Add batch dimension

pred_probs = neuronix.test_model(model=model_0,
                                X=data,
                                device=device)

print('Model Name: ', pred_probs['model_name'])
print('Predictions:')
print(class_names[pred_probs['predictions']])

cv.imshow('test', cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR))
cv.waitKey(0)
cv.destroyAllWindows()