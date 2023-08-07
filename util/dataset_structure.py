import h5py, math, torch, fnmatch, os
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from XMLHandler import XMLHandler

def digitize(tensor, bin_edges,device, middle='true'):
    bin_edges = torch.tensor(bin_edges,device=torch.device(device))

    # Digitize the tensor into bin indices
    bin_indices = torch.bucketize(tensor, bin_edges)
    
    # Calculate the corresponding middle values
    bin_indices[bin_indices >= len(bin_edges)] = len(bin_edges)-1
    bin_indices[bin_indices == -1] = 0
    middle_values = (bin_edges[bin_indices] + bin_edges[bin_indices - 1]) / 2
    return middle_values

def digitize_input(sample_list, particle, filename):
    xml = XMLHandler(particle, filename=filename)
    r_edge = xml.r_edges[0]
    n_theta_bin = xml.a_bins[0]
    theta_edge = np.linspace(-math.pi, math.pi, n_theta_bin)
    z_bin = len(xml.r_edges)
    z_edge = np.linspace(-0.5, z_bin - 0.5,z_bin+1)
    trans_event = []
    for event in sample_list:
        r = torch.sqrt(event[:,1]*event[:,1] + event[:,2]*event[:,2])
        theta = torch.atan(event[:,2]/event[:,1]) + torch.pi*torch.where(event[:,1]<0, torch.tensor(1), torch.tensor(0))
        middle_r = digitize(r,r_edge,'cpu')
        middle_theta = digitize(theta, theta_edge,'cpu')
        middle_z = digitize(event[:,3],z_edge,'cpu')
        x = middle_r*torch.cos(middle_theta)
        y = middle_r*torch.sin(middle_theta)
        output_ = torch.stack((event[:,0],x,y,middle_z),dim=1)
        trans_event.append(output_)
    return trans_event


class cloud_dataset(Dataset):
  def __init__(self, filename, transform=None, transform_y=None, device='cpu', transformed=False):
    loaded_file = torch.load(filename, map_location=torch.device(device))
    self.data = loaded_file[0]
    if transformed:
      self.condition = torch.tensor(loaded_file[1], device=torch.device(device))
    else:
      self.condition = torch.tensor(loaded_file[1], dtype=torch.long, device=torch.device(device))
    self.transform = transform
    self.transform_y = transform_y
    self.min_y = torch.min(self.condition)
    self.max_y = torch.max(self.condition)
    self.max_nhits = -1
    self.device = device
    self.filename = filename

  def __getitem__(self, index):
    x = self.data[index]
    y = self.condition[index]
    if self.transform:
        x = self.transform(x,y,self.device)
    if self.transform_y:
       y = self.transform_y(y, self.min_y, self.max_y)
    return x,y
  
  def __len__(self):
    return len(self.data)

  def digitize(self, particle='electron', xml_bin='binning_dataset_2.xml'):
    self.data = digitize_input(self.data, particle, xml_bin)

  def transformed(self):
    transformed_showers = []
    xmin, xmax = -40, 40
    ymin, ymax = -40, 40
    zmin, zmax = 0, 45
    e_middle, e_rms = -10 ,5 #Experimental value
    for idx, showers in enumerate(self.data):
      #e_ = torch.log(0.5*(showers[:,0]/self.condition[idx]+1.)/(1.-0.5*(self.condition[idx]+1.)))
      e_ = showers[:,0]/(self.condition[idx]*2.)
      e_ = (torch.log(e_/(1.-e_))-e_middle)/e_rms
      x_ = 2.*(showers[:,1] - xmin)/(xmax-xmin)-1.
      y_ = 2.*(showers[:,2] - ymin)/(ymax-ymin)-1.
      z_ = 2.*(showers[:,3] - zmin)/(zmax-zmin)-1.
      transformed_showers.append(torch.stack((e_,x_,y_,z_),-1))
    self.data = transformed_showers
    self.condition = self.condition/torch.max(self.condition)
    self.condition.to(torch.float32)


  def padding(self, value = -20):
    
    for showers in self.data:
      if len(showers) > self.max_nhits:
        self.max_nhits = len(showers)

    print(self.max_nhits)
    padded_showers = []
    for showers in self.data:
      pad_hits = self.max_nhits-len(showers)
      if len(showers) == 0: continue
      padded_shower = F.pad(input = showers, pad=(0,0,0,pad_hits), mode='constant', value = value)
      padded_showers.append(padded_shower)

    self.data = padded_showers

  def save(self, save_name = None):
    if save_name is None:
      print("Must assign name to saved file.")
      return 0
    print(self.condition.dtype)
    torch.save([self.data, self.condition], save_name)

class rescale_conditional:
  '''Convert hit energies to range |01)
  '''
  def __init__(self):
            pass
  def __call__(self, conditional, emin, emax):
     e0 = conditional
     u0 = (e0-emin)/(emax-emin)
     return u0

class rescale_energies:
        '''Convert hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition, device='cpu'):
            #Eprime = features.x[:,0]/(2*condition)
            Eprime = features[:,0]/(2*condition)
            alpha = 1e-06
            x = alpha+(1-(2*alpha))*Eprime
            rescaled_e = torch.tensor([math.log( x_/(1-x_) ) if (x_ > 0 ) else -20 for x_ in x], device=torch.device(device))
#            rescaled_e = x
            #x_ = features.x[:,1]
            x_ = features[:,1]
            #y_ = features.x[:,2]
            y_ = features[:,2]
            #z_ = features.x[:,3]
            z_ = features[:,3]
            
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            #self.features = Data(x=stack_)
            self.features = stack_
            
            return self.features

class unscale_energies:
        '''Undo conversion of hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition):
            rescaled_e = features[:,0]*(condition)
            x_ = features[:,1]
            y_ = features[:,2]
            z_ = features[:,3]
            
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            self.features = Data(x=stack_)
            
            return self.features

class VESDE:
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000, device='cuda'):
    """Construct a Variance Exploding SDE.
    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.N = N

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x, device=x.device)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
