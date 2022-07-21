from copy import deepcopy
from pyexpat import model
import torch
import numpy as np
from PIL import Image
from EA_components_OhGreat.Population import Population


class LogCrossentropy:
    def __init__(self, min, init_img, true_label, epsilon, model, batch_size, device):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.img_shape = (3,299,299)
        self.true_label = true_label
        self.min = min
        self.device = device
        self.orig_img = init_img
        # this is the processed image to be added to the generated noise
        self.orig_img_norm = torch.unsqueeze((torch.tensor(
                                            np.array(self.orig_img)
                                            )/255.
                                            ).permute((2,0,1)), 
                                            dim=0)

    def worst_eval(self):
        """ Return worst evaluation possible for the current problem configutation.
        """
        return np.inf if self.min else -np.inf

    def __call__(self, X: Population):
        ret_vals = []
        # clip individuals to epsilon interval
        X.individuals = X.individuals.clip(-self.epsilon,self.epsilon)
        # reshape to match population and image shape
        inds = torch.tensor(X.individuals.reshape((X.pop_size, *self.img_shape)))
        # create noise + original image attacks
        solutions = (torch.add(self.orig_img_norm, inds).clip(0,1)*255).type(torch.uint8)
        # pass through the model's preprocessing
        solutions = self.model.transforms(solutions)

        curr_best_eval = self.worst_eval()
        # evaluate solutions through model
        with torch.no_grad():
            for sol in solutions:
                pred = self.model.simple_eval(sol.unsqueeze(dim=0))
                curr_eval = torch.log(pred[:, self.true_label]).item()
                if (self.min and pred[:, self.true_label] < curr_best_eval) or (not self.min and pred[:, self.true_label] > curr_best_eval):
                    curr_best_eval = pred[:, self.true_label]
                ret_vals.append(curr_eval)
        # update individuals' fitness
        X.fitnesses = np.array(ret_vals)
        print("Current best confidence:",curr_best_eval.item())
        

