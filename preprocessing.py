import torch
from torchvision import transforms

# https://en.wikipedia.org/wiki/Relative_luminance#Relative_luminance_and_%22gamma_encoded%22_colorspaces
weights_luminance = torch.tensor([0.2126, 0.7152, 0.0722])
p = transforms.Resize(size=84)
trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()

def preprocessing(previous_state, state):
    """Reduce the size of input to reduce computation
    From "Human-level control through deep reinforcement learning", §METHODS
    - Take maximum value for each pixel colour value over the frame being encoded 
      and the previous frame. This was necessary to remove flickering that is 
      present in games where some objects appear only in even frames while other
      objects appear only in odd frames,
    - Extract the Y channel, also known as luminance
    - Rescale to 84x84

84 3 84.
    Args:
        previous_state ([type]): [description]
        state ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Remove flickering by taking the maximum value of each pixel colour
    state = torch.max(torch.stack([previous_state, state]), dim=0).values
    # crop the image, we don't need the header and the score bar
    state = state[20:-50,:,:]
    # Resize to 84x84
    state = trans1(p(trans(state.permute(2, 0, 1)))).permute(1, 2, 0)
    # Extract luminance (https://en.wikipedia.org/wiki/Relative_luminance)
    state = (state * weights_luminance).sum(-1)
    return state