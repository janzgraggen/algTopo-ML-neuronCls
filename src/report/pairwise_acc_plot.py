# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# Labels
labels = ['G', 'PI_CNN', 'W', 'B', 'SW', 'LS', 'M']

# Create and fill matrix
placeholder_matrix = np.zeros((7, 7))
upper_tri_values = [
    [0.84, 0.86, 0.85, 0.89, 0.84, 0.92, 0.86],
    [0, 0.83, 0.84, 0.86, 0.84, 0.88, 0.82],
    [0, 0, 0.82, 0.86, 0.81, 0.90, 0.82],
    [0, 0, 0, 0.84, 0.85, 0.90, 0.82],
    [0, 0, 0, 0, 0.70, 0.91, 0.71],
    [0, 0, 0, 0, 0, 0.89, 0.91],
    [0, 0, 0, 0, 0, 0, 0.72]
]

for i in range(7):
    for j in range(i + 1, 7):
        placeholder_matrix[i][j] = upper_tri_values[i][j]
for i in range(7):
    placeholder_matrix[i][i] = upper_tri_values[i][i]

# Mask lower triangle
mask = np.tril(np.ones_like(placeholder_matrix, dtype=bool), -1)

# Set up plot
plt.figure(figsize=(8, 6))
ax = sns.heatmap(placeholder_matrix, xticklabels=labels, yticklabels=labels,
                 annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, mask=mask)

# Red frame for important upper-triangular combinations

# Semi-transparent gray overlay on diagonal cells
for i in range(7):
    ax.add_patch(patches.Rectangle((i, i), 1, 1, color='black', alpha=0.6, lw=0))

inset = 0.1  # How much to inset the square (in cell units)
size = 1 - 2 * inset
for i in range(7):
    for j in range(i + 1, 7):
        if placeholder_matrix[i][j] > placeholder_matrix[i][i] and placeholder_matrix[i][j] > placeholder_matrix[j][j]:
            ax.add_patch(patches.Rectangle((j + inset, i + inset), size, size,
                                           fill=False, edgecolor='red', lw=2))
            
# Add a legend patch directly inside the lower-left corner of the heatmap
legend_text = 'Fusion improves over both individual embeddings'
ax.text(0.5, 6.325, legend_text, fontsize=9, color='black', ha='left', va='top')

# Add a red inset square as visual cue near the text
ax.add_patch(patches.Rectangle((0.2, 6.3), 0.2, 0.2, fill=False, edgecolor='red', lw=2))

# Label axes
plt.xlabel("Embeddings")
plt.ylabel("Embeddings")
plt.tight_layout()
plt.show()
