import json

import matplotlib.pyplot as plt

with open('toy_results.txt') as f:
    results = json.load(f)

dims = [2, 5, 10, 20, 50, 100, 150, 200, 300]
plt.plot(dims,results['F-MCVAE'], label='F-MCVAE')
plt.plot(dims, results['VAE'], label='VAE')
plt.plot(dims, results['IWAE'], label='IWAE')
plt.plot(dims, results['A-MCVAE'], label='A-MCVAE')
plt.plot(dims, results['L-MCVAE'], label='L-MCVAE')
plt.plot(dims, results['RealNVP'], label='RealNVP')
# ... plot others ...
plt.xlabel("Latent Dimension")
plt.ylabel("Discrepancy")
plt.legend()
plt.show()