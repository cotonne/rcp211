import torch
from torchvision import transforms

weights_luminance = torch.tensor([0.2126, 0.7152, 0.0722])
p = transforms.Resize(size=84)
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

def preprocessing(previous_state, state):
    # Remove flickering by taking the maximum value of each pixel colour
    state = torch.max(torch.stack([previous_state, state]), dim=0).values
    # crop the image, we don't need the header and the score bar
    state = state[20:-50,:,:]
    # Resize to 84x84
    state = trans1(p(trans(state.permute(2, 0, 1)))).permute(1, 2, 0)
    # Extract luminance (https://en.wikipedia.org/wiki/Relative_luminance)
    state = (state * weights_luminance).sum(-1)
    return state