#!/usr/bin/env python
"""
Main script to run Bayesian KAN examples
Choose which inference method to run
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Run Bayesian KAN with different inference methods'
    )
    parser.add_argument(
        'method',
        type=str,
        choices=['vi', 'mcmc', 'laplace', 'compare', 'all'],
        help='Inference method to run: vi (Variational Inference), '
             'mcmc (MCMC Sampling), laplace (Laplace Approximation), '
             'compare (Compare all methods), all (Run all examples)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Bayesian Kolmogorov-Arnold Networks")
    print("="*70)
    
    if args.method == 'vi' or args.method == 'all':
        print("\n>>> Running Variational Inference example...")
        from .examples import example_vi
        example_vi.main()
    
    if args.method == 'mcmc' or args.method == 'all':
        print("\n>>> Running MCMC Sampling example...")
        from .examples import example_mcmc
        example_mcmc.main()
    
    if args.method == 'laplace' or args.method == 'all':
        print("\n>>> Running Laplace Approximation example...")
        from .examples import example_laplace
        example_laplace.main()
    
    if args.method == 'compare':
        print("\n>>> Running Comparison of all methods...")
        from .examples import example_comparison
        example_comparison.main()
    
    print("\n" + "="*70)
    print("Execution complete!")
    print("="*70)


if __name__ == "__main__":
    main()