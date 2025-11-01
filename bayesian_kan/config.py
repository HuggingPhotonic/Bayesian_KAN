"""
Bayesian KAN Configuration
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


DEVICE = get_device()