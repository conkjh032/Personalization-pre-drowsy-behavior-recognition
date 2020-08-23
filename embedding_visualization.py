import gc
import torch
import numpy as np

from siamese_net import EmbeddingNet
from data_loader import Data_loader
from matplotlib import pyplot as plt

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model path
checkpoints_path = 'checkpoints/model_epoch_100.pth'
# data path
dataset_path = 'visualization_data'

model = EmbeddingNet()
model.to(device)

# load model
checkpoint = torch.load(checkpoints_path)
model_parameters = checkpoint['model_state_dict']
model.load_state_dict(model_parameters)
model.eval()
gc.collect()

# load data
data_loader = Data_loader(dataset_path=dataset_path)
data_loader.split_train_datasets()
images, labels = data_loader.visualization_data_load()
images = np.transpose(images, [0, 3, 2, 1])

# plot data on graph to check how model works well
def plot_embeddings(embeddings, targets):

	for i in [0, 1]:
		inds = np.where(targets == i)[0]
		x = embeddings[inds, 0]
		y = embeddings[inds, 1]
		plt.scatter(x, y, alpha = 0.5, color = colors[i])
	plt.legend(classes)
	plt.show()

images = torch.from_numpy(images)
labels = torch.from_numpy(np.array(labels))
images = images.to(device)
embeddings = model.get_embeddings(images)
plot_embeddings(embeddings.detach().cpu().numpy(), labels.numpy())

