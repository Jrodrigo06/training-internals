import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/training_log_dead_neuron.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(data['epoch'], data['loss'])
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_yscale('log')  

axes[0, 1].plot(data['epoch'], data['w'], label='w')
axes[0, 1].plot(data['epoch'], data['b'], label='b')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Parameter Value')
axes[0, 1].set_title('Parameters (w, b)')
axes[0, 1].legend()

axes[1, 0].plot(data['epoch'], data['grad_norm'])
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Gradient Norm')
axes[1, 0].set_title('Gradient Magnitude')
axes[1, 0].set_yscale('log')

axes[1, 1].plot(data['w'], data['b'])
axes[1, 1].scatter(data['w'].iloc[0], data['b'].iloc[0], c='green', s=100, label='Start', zorder=5)
axes[1, 1].scatter(data['w'].iloc[-1], data['b'].iloc[-1], c='red', s=100, label='End', zorder=5)
axes[1, 1].set_xlabel('w')
axes[1, 1].set_ylabel('b')
axes[1, 1].set_title('Parameter Space Trajectory')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('data/training_curves_dead_neuron.png', dpi=150)
print("Saved data/training_curves.png")
plt.show()