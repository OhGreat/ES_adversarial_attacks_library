from copy import deepcopy
import torch
import numpy as np
from scipy.ndimage import zoom
from torchvision.transforms import Resize
from EA_numpy.Population import Population
from Models import GenericModel
from PIL import Image


class LogCrossentropy:
    def __init__(self, min, atk_mode, init_img: Image, label: int, epsilon: float, 
                downsample: float, model: GenericModel, batch_size: int, device: str):
        self.device = device
        self.model = model
        self.epsilon = epsilon
        self.downsample = downsample
        self.img_shape = self.model.input_shape
        self.label = label
        self.min = min
        self.atk_mode = atk_mode
        self.orig_img = init_img.resize(model.transf_shape[1:])
        # this is the processed image to be added to the generated noise
        self.orig_img_norm = torch.unsqueeze((torch.tensor(
                                            np.array(self.orig_img)
                                            )/255.).permute((2,0,1)),dim=0)
        # create downsampled image if required
        if downsample is not None:
            curr_img = zoom(np.array(self.orig_img), 
                            zoom=(downsample,downsample,1), 
                            order=1)
            print("Downsampled image shape:", curr_img.shape)
        else: curr_img = np.array(self.orig_img)
        self.curr_img_shape = curr_img.shape

        # define individual shape
        if atk_mode == "1D_one-pixel":  # 1D one pixel attack
            self.ind_size = 4  # values: pixel value, x, y, channel
        elif atk_mode == "3D_one-pixel": # 3D one pixel attack
            self.ind_size = 5
        elif atk_mode == "R_channel" or atk_mode == "shadow_noise":
            self.ind_size = np.prod(curr_img.shape[:2])
        elif atk_mode == "all_channels":  # all channels attack
            self.ind_size = np.prod(curr_img.shape)
        else:
            exit("Select a valid attack method.")
        print("Problem dimension (individual size):", self.ind_size)

    def worst_eval(self):
        """ Return worst evaluation possible for the current problem configutation.
        """
        return np.inf if self.min else -np.inf

    def __call__(self, X: Population, ret_sol=False):

        if self.atk_mode == "1D_one-pixel":  # 1D one pixel attack
            """ Individual representation is a vector of 4 values:
                    val in pos 0: pixel noise
                    val in pos 1, 2: coordinates on image
                    val in pos 3: channel
            """
            inds = torch.tensor(X.individuals)
            # fix coordinates
            inds[:,1] = (inds[:,1].clip(0,1) * self.orig_img_norm.shape[-2]-1)
            inds[:,2] = (inds[:,2].clip(0,1) * self.orig_img_norm.shape[-1]-1)
            # fix channel
            inds[:,-1][inds[:,-1] < 0.33] = 0
            inds[:,-1][inds[:,-1] >= 0.66] = 2
            inds[:,-1][(inds[:,-1] >= 0.33) & (inds[:,-1] < 0.66)] = 1
            # create solution images
            solutions = torch.repeat_interleave(self.orig_img_norm, repeats=X.pop_size,dim=0)
            for curr_ind in range(X.pop_size):
                # get coordinates and channel
                x = inds[curr_ind][1].type(torch.int)
                y = inds[curr_ind][2].type(torch.int)
                channel = inds[curr_ind][-1].type(torch.int)
                # add noise to current solution
                solutions[curr_ind,channel,x,y] = inds[curr_ind][0]
            # fix pixel boundaries
            solutions = (solutions.clip(0,1)*255).type(torch.uint8)
        
        elif self.atk_mode == "3D_one-pixel": # 3D one pixel attack
            """ Individual representation is a vector of 5 values:
                    val in pos 0,1,2: pixel noise for each channel
                    val in pos 3, 4: coordinates on image
            """
            inds = torch.tensor(X.individuals)
            # fix coordinates
            inds[:,3] = (inds[:,3].clip(0,1) * self.orig_img_norm.shape[-2]-1)
            inds[:,4] = (inds[:,4].clip(0,1) * self.orig_img_norm.shape[-1]-1)
            # create starting solutions
            solutions = torch.repeat_interleave(self.orig_img_norm, repeats=X.pop_size,dim=0)
            # add noise to solutions
            for curr_ind in range(X.pop_size):
                #get coordinates
                x = inds[curr_ind][3].type(torch.int)
                y = inds[curr_ind][4].type(torch.int)
                # add noise to each solution
                solutions[curr_ind,0:3,x,y] = inds[curr_ind][0:3]
            # fix pixel boundaries
            solutions = (solutions.clip(0,1)*255).type(torch.uint8)

        elif self.atk_mode == "R_channel":  # noise on red channel
            # clip individuals to epsilon interval
            inds = torch.tensor(X.individuals).clip(-self.epsilon,self.epsilon)

            # reshape to match population and image shape
            inds = inds.reshape((X.pop_size, 1, *self.curr_img_shape[:2]))
            if self.downsample is not None:
                inds = Resize(size=self.orig_img_norm.shape[2:]).forward(inds)
            # add noise to first channel
            solutions = torch.add(self.orig_img_norm[:,0,:], inds)
            # concatenate the other two channels to our solutions,
            # clip values and transform values in integers from 0 to 255 
            solutions = torch.stack([((torch.cat((solutions[i], self.orig_img_norm[0,1:,:]), dim=0)
                                    ).clip(0,1)*255).type(torch.uint8)
                                        for i in range(len(solutions))])
        
        elif self.atk_mode == "shadow_noise":  # apply noise as shadow on all channels
            # clip individuals to epsilon interval
            inds = torch.tensor(X.individuals).clip(-self.epsilon,self.epsilon)
            # reshape to match population and image shape
            inds = inds.reshape((X.pop_size, 1, *self.curr_img_shape[:2])) 
            if self.downsample is not None:
                inds = Resize(size=self.orig_img_norm.shape[2:]).forward(inds)
            # add same noise to all channels
            solutions = torch.stack([torch.add(inds[i],self.orig_img_norm[0,:]) for i in range(len(inds))])
            # clip image, multiply by 255 and take integer values
            solutions = (solutions.clip(0,1)*255).type(torch.uint8)

        elif self.atk_mode == "all_channels":  # noise on all channels
            # clip individuals to epsilon interval
            inds = torch.tensor(X.individuals).clip(-self.epsilon,self.epsilon)
            # reshape to match population and image shape
            inds = inds.reshape((X.pop_size, 3, *self.curr_img_shape[:2]))
            if self.downsample is not None:
                inds = Resize(size=self.orig_img_norm.shape[2:]).forward(inds)
            # create noise + original image attacks
            solutions = (torch.add(self.orig_img_norm, inds).clip(0,1)*255).type(torch.uint8)

        else:
            exit('Please choose a correct attack modality.')

        # make a copy if we need to return solution later
        if ret_sol: sol_copy = deepcopy(solutions)

        # apply model transformations to noised image attacks
        solutions = self.model.transforms(solutions)

        fitnesses = []
        # temporary to print best accuracy of generation
        curr_best_eval = self.worst_eval()
        # evaluate solutions through model
        with torch.no_grad():
            for sol in solutions:
                pred = self.model.simple_eval(sol.unsqueeze(dim=0).to(self.device))
                # fitness is the logarithm of the predicted confidence
                curr_eval = torch.log(pred[:, self.label]).item()
                fitnesses.append(curr_eval)
                # check best found confidence of generation for print
                if (self.min and pred[:, self.label] < curr_best_eval) or (
                    not self.min and pred[:, self.label] > curr_best_eval):
                    curr_best_eval = pred[:, self.label]
        # update individual fitnesses
        X.fitnesses = np.array(fitnesses)
        
        if ret_sol: return sol_copy
        else:
            print("Current best confidence:",curr_best_eval.item())