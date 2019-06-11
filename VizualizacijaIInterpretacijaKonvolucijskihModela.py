import io

from PIL import Image
import requests

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import cv2
import random

class VggVisualizer:
	def __init__(self):
		LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
		self.labels = {int(key): value for key, value in requests.get(LABELS_URL).json().items()}
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.preprocess = transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
			])
		self.model = models.vgg16(pretrained=True).to(self.device).eval()
		
	def getAndPreprocessImage(self, image_url):
		image = Image.open(io.BytesIO(requests.get(image_url).content))
		image = self.preprocess(image)
		image = Variable(image.unsqueeze(0)).to(self.device)
		return image
		
	def convertToGrayByMax(self, image):
		gray = torch.zeros([224, 224]).to(self.device)
		for i in range(224):
			for j in range(224):
				gray[i][j] = max([image[0][i][j], image[1][i][j], image[2][i][j]])
				
		gray = gray.unsqueeze(dim = 0)
		return gray
		
	def generateNoiseImage(self):
		image = np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224)))/255
		image = torch.tensor(image, dtype=torch.float32)
		image = image.to(self.device)
		image = Variable(image)
		return image
		
	def classModelVisualization(self, labelIndex):
		image = self.generateNoiseImage()
		image.requires_grad = True
		
		object = self.labels[labelIndex]
		
		fig = plt.figure(1, figsize=(10, 6))
		fig.suptitle('class model visualization ' + object, fontsize=15)
		grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=(0.05, 0.3))

		lr = 0.05
		optimizer = optim.SGD([image], lr=lr, momentum=0.9)
		
		for i in range(1, 81):
			optimizer.zero_grad()
			self.model.zero_grad()
			prediction = self.model(image)
			#print(self.labels[prediction.argmax().cpu().item()])
			#print(F.softmax(prediction, dim=1)[0][labelIndex])
			#score = prediction[0][labelIndex]
			
			grad_map = torch.zeros(prediction.shape, dtype=torch.float).to(self.device)
			grad_map[0][labelIndex] = -1
			
			prediction.backward(grad_map)
			optimizer.step()
			#print(image.grad.data.shape)
			if i%20 == 0:
				grid[int(i/20-1)].set_title(str(i) + ' iterations')
				imshow(grid[int(i/20-1)], image, False)
			
		plt.show()
		
	def saliencyMap(self, image_url, gray=True):
		image = self.getAndPreprocessImage(image_url)
		image.requires_grad = True
		
		self.model.zero_grad()
		prediction = self.model(image)
		index = prediction.argmax().cpu().item()
		
		prediction[0][index].backward()
		
		fig = plt.figure(1, figsize=(10, 6))
		fig.suptitle('saliency map ' + self.labels[index], fontsize=15)
		grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=(0.05, 0.3)) 
		imshow(grid[0], image)
		
		data = image.grad.data
		
		if(gray):
			data = self.convertToGrayByMax(data.squeeze(dim=0))
			
		im1 = data.clamp(min=0)
		im2 = data.clamp(min=0.03)
		im3 = data.clamp(min=0.05)
		
		grid[1].set_title('clamp min=0')
		imshow(grid[1], im1, False, gray)
		grid[2].set_title('clamp min=0.03')
		imshow(grid[2], im2, False, gray)
		grid[3].set_title('clamp min=0.05')
		imshow(grid[3], im3, False, gray)
		plt.show()
		
	def guidedBackpropagation(self, image_url, gray=False):
		self.newImg = None
		self.fmaps = []
		
		def first_layer_hook_fn(module, grad_out, grad_in):
			self.newImg = grad_out[0] 
			
		def forward_hook_fn(module, input, output):
			self.fmaps.append(output)
			
		def backward_hook_fn(module, grad_out, grad_in):
			new_grad_out = grad_out[0].clamp(min=0)
			fgrad = self.fmaps[-1]
			fgrad[fgrad > 0] = 1
			new_grad_out = new_grad_out * fgrad
			del self.fmaps[-1]
			
			return (new_grad_out,)
            
		modules = list(self.model.features._modules.items())

		hooks = []
		for name, module in modules:
			if isinstance(module, nn.ReLU):
				hooks.append(module.register_forward_hook(forward_hook_fn))
				hooks.append(module.register_backward_hook(backward_hook_fn))

		fLayer = modules[0][1] 
		hooks.append(fLayer.register_backward_hook(first_layer_hook_fn))

		image = self.getAndPreprocessImage(image_url)
		image.requires_grad = True
		
		self.model.zero_grad()
		output = self.model(image)

		index = output.argmax().cpu().item()
		output[0][index].backward()
		
		if(gray):
			self.newImg = self.convertToGrayByMax(self.newImg.squeeze(dim=0))
			self.newImg = self.newImg.clamp(min=0)
		
		fig = plt.figure(1, figsize=(10, 6))
		fig.suptitle('guided backpropagation ' + self.labels[index], fontsize=15)
		grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=(0.05, 0.05)) 
		imshow(grid[0], image)
		imshow(grid[1], self.newImg, False, gray)
		plt.show()
		
		for hook in hooks:
			hook.remove()
		
		
	def classActivationMap(self, image_url):
		self.activations = None
		self.gradients = None
		
		def hook_fn(module, input, output):
			self.activations = output
		
		def bcw_hook(module, grad_out, grad_in):
			self.gradients = grad_in[0]
		
		last_conv = self.model._modules['features'][29]
		hook_fw = last_conv.register_forward_hook(hook_fn)
		hook_bck = last_conv.register_backward_hook(bcw_hook)
		
		image = self.getAndPreprocessImage(image_url)
		output = self.model(image)
		
		topres = torch.topk(output, 4)
		scores = topres[0].cpu().detach().numpy()[0]
		indexes = topres[1].cpu().detach().numpy()[0]
		
		fig = plt.figure(1, figsize=(10, 6))
		fig.suptitle('class activation map, top 4', fontsize=15)
		grid = ImageGrid(fig, 111, nrows_ncols=(4, 2), axes_pad=(2, 0.3))
		
		for i in range(4):
			output = self.model(image)
			self.model.zero_grad()
			output[:, indexes[i]].backward()
			pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

			for j in range(512):
				self.activations[:, j, :, :] *= pooled_gradients[j]
				
			heatmap = torch.mean(self.activations, dim=1).squeeze()

			heatmap = heatmap.clamp(min=0)
			heatmap /= torch.max(heatmap)
			#plt.imshow(heatmap)
				
			#image = Image.open(io.BytesIO(requests.get(image_url).content)).convert('RGB')
			#image = np.array(image)
			imageAlt = unNormalizeImage(image.cpu().detach().squeeze().numpy().transpose((1, 2, 0)))*255
			imageAlt = imageAlt.astype(int)
			heatmap = cv2.resize(heatmap.cpu().detach().numpy(), (imageAlt.shape[1], imageAlt.shape[0]))
			#heatmap[heatmap < 0.6] = 0
			heatmap = 255-np.uint8(255*heatmap)
			heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
			grid[2*i].set_title('Heatmap for: ' + self.labels[indexes[i]])
			grid[2*i].imshow(heatmap)
			
			superimposed_img = heatmap*0.4 + imageAlt
			superimposed_img = superimposed_img.astype(int)

			grid[2*i+1].set_title(self.labels[indexes[i]])
			grid[2*i+1].imshow(superimposed_img)
		plt.show()
		hook_fw.remove()
		hook_bck.remove()
		
	def layerVisualization(self, layer, filter=None, lr=0.05, iterations=32):
		subnet = self.model._modules['features'][0:layer+1]
		
		image = self.generateNoiseImage()
		image.requires_grad = True;
		
		optimizer = optim.Adam([image], lr=lr, weight_decay=1e-6)
		for i in range(iterations):
			optimizer.zero_grad()
			self.model.zero_grad()
			output = subnet(image)
			if(filter is None):
				output = output[0]
				loss = -torch.mean(output).squeeze()
			else:
				output = output[0][filter]
				loss = -torch.mean(output)
			loss.backward()
			optimizer.step()
			
		image = image.cpu().detach().numpy().squeeze(0).transpose(1,2,0)
		return image
		#plt.imshow(image)
		#plt.show()
		
	def deepDream(self, layer, image_url, lr=0.2, iterations=30):
		output = None
		subnet = self.model._modules['features'][0:layer+1]
		
		image = self.getAndPreprocessImage(image_url)
		image.requires_grad = True
		
		optimizer = optim.SGD([image], lr=lr,  weight_decay=1e-4)
		for i in range(iterations):
			self.model.zero_grad()
			optimizer.zero_grad()
			output = subnet(image)
			#output = output.clamp(min=0)
			loss = -output.norm()
			loss.backward()
			optimizer.step()
			
		image = image.cpu().detach().numpy().squeeze(0).transpose(1,2,0)
		image = unNormalizeImage(image)
		return image
		
def unNormalizeImage(image):
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	original = std * image + mean
	return original
	
		

def imshow(axis, inp, unNormalize=True, gray=False):
	inp = inp.cpu().detach().numpy().squeeze(0)
	if not gray:
		inp = inp.transpose((1, 2, 0))
	if unNormalize:
		inp = unNormalizeImage(inp)

	if gray:
		axis.imshow(inp, cmap='gray')
	else:
		axis.imshow(inp)
		
def showConvLayers(vggVis):
	layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
	
	fig = plt.figure(1, figsize=(14, 10))
	fig.suptitle('convLayers visualization ' + str(layers), fontsize=15)
	grid = ImageGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0.05)

	i = 0
	for layer in layers:
		img = vggVis.layerVisualization(layer)
		grid[i].imshow(img)
		i += 1
	plt.show()
	
def showLayerFilters(vggVis, layer):
	noFilters = vggVis.model._modules['features'][layer].out_channels
	toShow = random.sample(range(noFilters), 15)
	
	fig = plt.figure(1, figsize=(14, 10))
	fig.suptitle('layer ' + str(layer) + ' random 15 filters visualization ' + str(toShow), fontsize=15)
	grid = ImageGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0.05)
	
	i = 0
	for index in toShow:
		img = vggVis.layerVisualization(layer, index)
		grid[i].imshow(img)
		i += 1
	plt.show()
	
def deepDreamConv(vggVis, image_url):
	layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
	fig = plt.figure(1, figsize=(14, 10))
	fig.suptitle('deep dream convLayers visualization ' + str(layers), fontsize=15)
	grid = ImageGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0.05)

	i = 0
	for layer in layers:
		img = vggVis.deepDream(layer, image_url, iterations=40)
		grid[i].imshow(img)
		i += 1
	plt.show()
	

CAT = 'https://s3.amazonaws.com/mlpipes/pytorch-quick-start/cat.jpg'
BUCKET = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwSDObrYZEeDYhkFy7Y6R8lr46y1xpCcY-1ICM-msbbJWY3vK_'
SHARK = 'https://upload.wikimedia.org/wikipedia/commons/5/56/White_shark.jpg'
IG = 'https://cdn-images-1.medium.com/max/500/1*3Lf6qSSgaBhjAZTOxkPvdw.png'
SKY = 'https://www.metoffice.gov.uk/binaries/content/gallery/metofficegovuk/hero-images/weather/cloud/cumulus-cloud.jpg'
UN = 'https://static.interestingengineering.com/images/APRIL/sizes/black_hole_resize_md.jpg'
ML = 'https://povijest.hr/wp-content/uploads/2016/08/lisa.jpg'
#543 dumbell, 2 shark

v = VggVisualizer()
v.classModelVisualization(543)
v.saliencyMap(ML)
v.guidedBackpropagation(ML)
v.classActivationMap(ML)
#v.layerVisualization(22)
#v.layerVisualization(22, 11)
showConvLayers(v)
showLayerFilters(v, 5)
deepDreamConv(v, ML)