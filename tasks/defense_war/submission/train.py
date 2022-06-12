"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append("../../../")
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset
import importlib.util

from torchvision.transforms.functional import gaussian_blur

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class VirtualModel:
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model

    def preprocess(self, original_images):
        # image = torch.unsqueeze(original_images, 0)
        return original_images

    def detect_attack(self, original_image):
        # return true if original_image is an adversarial example; return false if original_image is benign.
        return False

    def get_batch_output(self, images, with_preprocess=True, skip_detect=False):
        outputs = []
        detect_outputs = []
        for ini_image in images:
            image = torch.unsqueeze(ini_image, 0)
            # detect funtion
            if (skip_detect != True) & (self.detect_attack(image) == True):
                outputs.append(torch.tensor([0,0,0,0]).to(self.device))
                detect_outputs.append(1)
            else:
                if with_preprocess == True:
                    image = self.preprocess(image)
                output = self.model(image).to(self.device)
                outputs.append(output[0])
                detect_outputs.append(0)
        outputs = torch.stack(outputs)
        detect_outputs = torch.tensor(detect_outputs).to(self.device)
        return outputs, detect_outputs

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


class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """

    """
    def __init__(self, device, file_path, target_label=None, epsilon=0.3, min_val=0, max_val=1):
        sys.path.append(file_path)
        from predict import LeNet
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.device = device
        self.min_val = min_val
        self.max_val = max_val
        self.target_label = target_label
        self.perturb = self.load_perturb("../attacker_list/target_FGSM")

    def load_perturb(self, attack_path):
        spec = importlib.util.spec_from_file_location('attack', attack_path + '/attack.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # for attack methods evaluator, the Attack class name should be fixed
        attacker = foo.Attack(VirtualModel(self.device, self.model), self.device, attack_path)
        return attacker


    def train(self, trainset, valset, device, epoches=30):
        self.model.to(device)
        self.model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                adv_inputs, _ = self.perturb.attack_batch(inputs, labels.detach().cpu())
                adv_inputs = torch.tensor(adv_inputs).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))
        return
    """
    def __init__(self, device, file_path, epsilon=0.3, min_val=0, max_val=1):
        sys.path.append(file_path)
        from predict import LeNet
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.optimizer = None
        self.alpha = 10/255

    def perturb(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        #loss = F.nll_loss(outputs, labels)
        loss = nn.NLLLoss()(nn.LogSoftmax(dim=1)(outputs), labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        return perturbed_image

    def perturb_FFGSM(self, images, labels, device):
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        loss = nn.CrossEntropyLoss()

        adv_images = images + torch.randn_like(images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.model(adv_images)

        # Calculate loss
        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def perturb_PGD(self, images, labels, device):
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(40):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def _do_iter(self, images, labels, device):
        X = images.to(device)
        Y = labels.to(device)

        X_adv = self.perturb_PGD(X, Y, device)

        logits_adv = self.model(X_adv)
        cost = nn.CrossEntropyLoss()(logits_adv, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item()

    def train(self, trainset, valset, device, epoches=10):
        
        
        self.model.to(device)
        self.model.train()  
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.model(inputs)

                # 10. to offset bias against perturbed images
                loss = 5. * criterion(outputs, labels)
                blur = inputs.clone().detach()
                for _ in range(blur.shape[0]):
                    blur[_, :, :] = gaussian_blur(blur[_, :, :], kernel_size=(11, 11), sigma=1.0)

                loss = 5. * criterion(self.model(blur), labels)
                #loss = 5. * criterion(gaussian_blur(outputs, kernel_size=(11, 11), sigma=(0.1, 2.0)), labels)
                #loss += 10. * criterion(model(self.perturb(inputs, labels)), labels)

                # Emulate PGD
                #last = self.perturb(inputs, labels)
                for _ in range(10):
                    # loss += criterion(self.model(self.perturb(inputs, labels)), labels)
                    loss += criterion(self.model(self.perturb_FFGSM(inputs, labels, device)), labels)

                #    #loss += criterion(model(last), labels)
                    #last = self.perturb(inputs, labels)

                #  loss = loss / 2
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))

        return self.model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_training = Adv_Training(device, file_path='.')
    dataset_configs = {
                "name": "CIFAR10",
                "binary": True,
                "dataset_path": "../datasets/CIFAR10/student/",
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }

    dataset = get_dataset(dataset_configs)
    trainset = dataset['train']
    valset = dataset['val']
    testset = dataset['test']
    adv_training.train(trainset, valset, device)
    torch.save(adv_training.model.state_dict(), "defense_war-model.pth")


if __name__ == "__main__":
    main()
