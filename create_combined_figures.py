#!/usr/bin/env python3
"""
Create combined figures with subfigures for the paper.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

def create_combined_confusion_matrices():
    """Create combined confusion matrices figure with CTU-13 and Honeypot subfigures (stacked vertically)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Load CTU-13 confusion matrix
    ctu13_img = mpimg.imread('results/plots/Figures/ctu13_confusion_matrices.png')
    axes[0].imshow(ctu13_img)
    axes[0].axis('off')
    axes[0].set_title('(a) CTU-13 Dataset', fontsize=14, fontweight='bold', pad=10)
    
    # Load Honeypot confusion matrix
    honeypot_img = mpimg.imread('results/plots/Figures/honeypot_confusion_matrices.png')
    axes[1].imshow(honeypot_img)
    axes[1].axis('off')
    axes[1].set_title('(b) Honeypot Dataset', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig('results/plots/Figures/combined_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/plots/Figures/combined_confusion_matrices.png")
    plt.close()

def create_combined_roc_curves():
    """Create combined ROC curves figure with CTU-13 and Honeypot subfigures (side-by-side horizontally)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Load CTU-13 ROC curve
    ctu13_img = mpimg.imread('results/plots/Figures/ctu13_roc_curves.png')
    axes[0].imshow(ctu13_img)
    axes[0].axis('off')
    axes[0].set_title('(a) CTU-13 Dataset', fontsize=14, fontweight='bold', pad=10)
    
    # Load Honeypot ROC curve
    honeypot_img = mpimg.imread('results/plots/Figures/honeypot_roc_curves.png')
    axes[1].imshow(honeypot_img)
    axes[1].axis('off')
    axes[1].set_title('(b) Honeypot Dataset', fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig('results/plots/Figures/combined_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Created: results/plots/Figures/combined_roc_curves.png")
    plt.close()

def main():
    """Create all combined figures."""
    print("Creating combined figures...")
    create_combined_confusion_matrices()
    create_combined_roc_curves()
    print("\n✓ All combined figures created!")

if __name__ == '__main__':
    main()
