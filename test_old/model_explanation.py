import torch
import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Mock model for demonstration
class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # 32x32x32
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # 64x16x16
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate and move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MockModel().to(device)

# Define input size
input_size = (3, 64, 64)

# Display model summary
print("Model Summary using torchsummary:")
summary(model, input_size)

# Create a random input tensor
x = torch.randn(1, *input_size).to(device)

# Generating model graph with torchviz
print("\nGenerating model graph with torchviz...")
y = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("model_topology")  # Saves "model_topology.png" in the current directory

# Using TensorBoard to visualize the model architecture
print("\nLogging model graph to TensorBoard...")
writer = SummaryWriter("runs/mock_model")  # Specify log directory
writer.add_graph(model, x)
writer.close()

# Instructions to view the TensorBoard
print("\nTo visualize in TensorBoard, run the following command in your terminal:")
print("tensorboard --logdir=runs")
print("Then open the provided URL in your browser to view the model graph and more.")