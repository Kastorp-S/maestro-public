"""
The template for the students to predict the result.
Please do not change LeNet, the name of batch_predict and predict function of the Prediction.

"""

import torch
import numpy as np
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim

"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        #self.batch1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        #self.batch2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        #x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.batch2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 4)

    def forward(self, x) -> torch.tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Prediction():
    """
    The Prediction class is used for evaluator to load the model and detect or classify the images. The output of the batch_predict function will be checked which is the label.
    If the label is the same as the ground truth, it means you predict the image successfully. If the label is -1 and the image is an adversarial examples, it means you predict the image successfully. Other situations will be decided as failure.
    You can use the preprocess function to clean or check the input data are benign or adversarial for later prediction.
    """
    
    """
    def __init__(self, device, file_path):
        self.device = device
        self.model = self.constructor(file_path).to(device)

    def constructor(self, file_path=None):
        model = LeNet()
        if file_path != None:
            model.load_state_dict(torch.load(file_path+'/defense_project-model.pth', map_location=self.device))
        model.eval()
        return model

    def preprocess(self, original_images):
        perturbed_image = original_images.unsqueeze(0)
        return perturbed_image

    def get_batch_output(self, images):
        predictions = []
        # for image in images:
        predictions = self.model(images).to(self.device)
            # predictions.append(prediction)
        # predictions = torch.tensor(predictions)
        return predictions

    def get_batch_input_gradient(self, original_images, labels, lossf=None):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        if lossf is None:
            loss = F.nll_loss(outputs, labels)
        else:
            loss = lossf(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        return data_grad
    """

    ###
    def __init__(self, device, file_path):
        self.device = device
        self.model = self.constructor(file_path).to(device)

    def constructor(self, file_path=None):
        model = LeNet()
        if file_path != None:
            model.load_state_dict(torch.load(file_path+'/defense_project-model.pth', map_location=self.device))
        return model

    def preprocess(self, original_images):
        perturbed_image = original_images.unsqueeze(0)
        return perturbed_image

    def detect_attack(self, original_image):
        ## Function to detect whether it is an attack
        Uthreshold = 1.85
        Lthreshold = 0.52
        image = self.preprocess(original_image)
        yhat = self.model(image)
        # print(yhat.data)
        _,predicted = torch.max(yhat.data, 1)
        # print(predicted)
        # print("Bound: ", yhat.data[0][predicted])

        ## Multiple boundaries - min arg

        if Lthreshold < (yhat.data[0][predicted]) < Uthreshold:
            return True
        return False

    def get_batch_output(self, images):
        predictions = []
        for image in images:
            if self.detect_attack(image):
                predictions.append(-1)
            else:
                prediction = self.predict(image)
                predictions.append(prediction)
        predictions = torch.tensor(predictions).to(self.device)
        return predictions

    def predict(self, image):
        image = self.preprocess(image)
        output = self.model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted
    ###

