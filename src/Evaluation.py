import torch
import numpy as np
from EA_components_OhGreat.Population import Population


class LogCrossentropy:
    def __init__(self, min, atk_mode, init_img, label, epsilon, model, batch_size, device):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.img_shape = self.model.input_shape
        self.label = label
        self.min = min
        self.atk_mode = atk_mode
        self.device = device
        self.orig_img = init_img
        # this is the processed image to be added to the generated noise
        self.orig_img_norm = torch.unsqueeze((torch.tensor(
                                            np.array(self.orig_img)
                                            )/255.), 
                                            dim=0).permute((0,3,1,2))

    def worst_eval(self):
        """ Return worst evaluation possible for the current problem configutation.
        """
        return np.inf if self.min else -np.inf

    def __call__(self, X: Population):

        if self.atk_mode == 1:  # noise on red channel
            # clip individuals to epsilon interval
            X.individuals = X.individuals.clip(-self.epsilon,self.epsilon)
            # reshape to match population and image shape
            inds = torch.tensor(X.individuals.reshape((X.pop_size, *self.model.input_shape[1:])))
            # add noise to first channel
            solutions = torch.add(self.orig_img_norm[:,0,:], inds).unsqueeze(dim=1)
            # concatenate the other two channels to our solutions,
            # clip values and transform values in integers from 0 to 255 
            solutions = torch.stack([((torch.cat((solutions[i], self.orig_img_norm[0,1:,:]), dim=0)).clip(0,1)*255).type(torch.uint8)
                            for i in range(len(solutions))])

        elif self.atk_mode == 2:  # noise on all channels
            # clip individuals to epsilon interval
            X.individuals = X.individuals.clip(-self.epsilon,self.epsilon)
            # reshape to match population and image shape
            inds = torch.tensor(X.individuals.reshape((X.pop_size, *self.img_shape)))
            # create noise + original image attacks
            solutions = (torch.add(self.orig_img_norm, inds).clip(0,1)*255).type(torch.uint8)

        elif self.atk_mode == 3:  # apply noise as shadow
            # clip individuals to epsilon interval
            X.individuals = X.individuals.clip(-self.epsilon,self.epsilon)
            # reshape to match population and image shape
            inds = torch.tensor(X.individuals.reshape((X.pop_size, *self.model.input_shape[1:])))
            # add noise to all channels
            solutions = torch.stack([torch.add(inds[i],self.orig_img_norm[0,:]) for i in range(len(inds))])
            # clip image, multiply by 255 and take integer values
            solutions = (solutions.clip(0,1)*255).type(torch.uint8)

        elif self.atk_mode == 4:  # one-dimensional one pixel attack
            """ Individual representation is a vector of 4 values:
                    val in pos 0: pixel noise
                    val in pos 1, 2: coordinates on image
                    val in pos 3: channel
            """
            # inds = torch.tensor(X.individuals.reshape(X.pop_size, -1))
            inds = torch.tensor(X.individuals)
            # clip pixel noise
            # REMINDER: we cannot cast to torch.uint8 because we are dealing with coordinates in the next positions
            inds[:,0] = (inds[:,0].clip(0,1))
            # fix coordinates
            inds[:,1] = (inds[:,1].clip(0,1) * self.model.input_shape[-2]-1)
            inds[:,2] = (inds[:,2].clip(0,1) * self.model.input_shape[-1]-1)
            # fix channel
            inds[:,-1][inds[:,-1] < 0.33] = 0
            inds[:,-1][inds[:,-1] >= 0.66] = 2
            inds[:,-1][(inds[:,-1] >= 0.33) & (inds[:,-1] < 0.66)] = 1
            # create solution images
            solutions = np.repeat(self.orig_img_norm, repeats=X.pop_size, axis=0)
            for curr_ind in range(X.pop_size):
                x = inds[curr_ind][1].type(torch.int)
                y = inds[curr_ind][2].type(torch.int)
                channel = inds[curr_ind][-1].type(torch.int)
                solutions[curr_ind,channel,x,y] += inds[curr_ind][0] 
        else:
            exit('Please choose a correct attack modality.')

        # apply model transformations to noised image attacks
        solutions = self.model.transforms(solutions)

        fitnesses = []
        # temporary to print best accuracy of generation
        curr_best_eval = self.worst_eval()
        # evaluate solutions through model
        with torch.no_grad():
            for sol in solutions:
                pred = self.model.simple_eval(sol.unsqueeze(dim=0))
                # fitness is the logarithm of the predicted confidence
                curr_eval = torch.log(pred[:, self.label]).item()
                fitnesses.append(curr_eval)
                # check best found confidence of generation for print
                if (self.min and pred[:, self.label] < curr_best_eval) or (
                    not self.min and pred[:, self.label] > curr_best_eval):
                    curr_best_eval = pred[:, self.label]
        # update individual fitnesses
        X.fitnesses = np.array(fitnesses)
        print("Current best confidence:",curr_best_eval.item())
        
