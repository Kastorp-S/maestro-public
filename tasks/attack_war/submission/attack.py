from typing import List, Iterator, Dict, Tuple, Any, Type

import numpy as np
import torch
from copy import deepcopy

np.random.seed(1901)

class Attack:
    def __init__(
        self,
        vm, device, attack_path,
        epsilon = 0.2,
        min_val = 0,
        max_val = 1
    ):
        """
        args:
            vm: virtual model is wrapper used to get outputs/gradients of a model.
            device: system on which code is running "cpu"/"cuda"
            epsilon: magnitude of perturbation that is added
        """
        self.vm = vm
        self.device = device
        self.attack_path = attack_path
        self.epsilon = 0.1
        self.min_val = 0
        self.max_val = 1

    def attack(
        self, original_images: np.ndarray, labels: List[int], target_label = None,
    ):
        original_images = original_images.to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        target_labels = target_label * torch.ones_like(labels).to(self.device)
        perturbed_image = original_images

        # -------------------- TODO ------------------ #

        # Write your attack function here
        """
        perturbed_image = torch.from_numpy(perturbed_image)

        adv_images = perturbed_image.clone().detach()

        batch_size = len(original_images)
    
        # Starting at a uniformly random point

        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.l2_threshold, self.l2_threshold)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(40):

            data_grad = self.vm.get_batch_input_gradient(original_images, labels)
            data_grad = torch.FloatTensor(data_grad)

            grad_norms = torch.norm(data_grad.view(batch_size, -1), p=2, dim=1) + (1e-10)
            data_grad = data_grad / grad_norms.view(batch_size,1,1,1)
            adv_images = adv_images.detach() + 0.2 * data_grad

            eta = adv_images - perturbed_image
            eta_norms = torch.norm(eta.view(batch_size, -1), p=2, dim=1)
            factor = self.epsilon / eta_norms
            factor = torch.min(factor, torch.ones_like(eta_norms))
            eta = eta * factor.view(-1,1,1,1)
            adv_images = torch.clamp(perturbed_image + eta, min=0, max=1).detach()
            """

        """
        e = 5
        epsilon = min(e+4, int(1.25*e))
        alpha = 0.01

        desired_labels = torch.tensor([torch.tensor(1)]).to(self.device)

        for i in range(epsilon):
            gradient = self.vm.get_batch_input_gradient(perturbed_image, desired_labels)
            perturbed_image = perturbed_image - (alpha*gradient.sign())
            perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
            perturbed_image = perturbed_image.detach().clone()
        """

        data_grad = self.compute_gradient(original_images, labels)
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)

        # ------------------ END TODO ---------------- #

        adv_outputs, detected_output  = self.vm.get_batch_output(perturbed_image)
        final_pred = adv_outputs.max(1, keepdim=True)[1]
        correct = 0
        correct += (final_pred == target_labels).sum().item()
        return perturbed_image.cpu().detach().numpy(), correct
