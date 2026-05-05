# 🌿 Hybrid CNN-SVM for Variance Suppression in Low-Data Classification
This project implements a hybrid classification architecture that decouples deep feature representation from decision boundary formulation. By replacing the standard probabilistic Softmax head with a deterministic Linear SVM, the model significantly suppresses algorithmic variance in data-scarce (few-shot) regimes.

# 🏛️ Architecture Overview
The system operates in two distinct phases to achieve high mathematical stability under data starvation:  
1. Phase I: Deep Texture Representation
- Backbone: A ResNet-18 architecture is used strictly as a high-dimensional feature extractor.
  
- Decoupling: The standard fully connected layers and Softmax function are removed.
  
- Latent Space: Input images ($256 \times 256$) are mapped into a 512-dimensional latent embedding vector ($v \in \mathbb{R}^{512}$).

2. Phase II: Maximum-Margin Geometric Fusion
- Normalization: Latent vectors undergo Isotropic Normalization (Z-score scaling) to ensure asymmetric feature magnitudes do not distort the hyperplane.  

- Classifier: A linear Support Vector Machine (SVM) optimized via L2-regularized hinge loss defines a rigid hyperplane between class clusters.  

- Benefit: Unlike cross-entropy, which overfits to local data density, the SVM enforces a geometric margin that guarantees the widest possible buffer zone between classes.

# 📊 Performance & Variance Suppression
The hybrid architecture was benchmarked using the Plant Village dataset (38 foliar disease classes) under strict $N$-shot constraints.  

Key Experimental Results:

| Samples per Class ($N$) | Baseline ResNet-18 Accuracy | **Hybrid CNN-SVM Accuracy** |
| :--- | :--- | :--- |
| 10 | $55.12\% \pm 6.81\%$ | **$61.45\% \pm 2.15\%$** |
| 25 | $67.38\% \pm 4.39\%$ | **$72.90\% \pm 1.03\%$** |
| 50 | $78.52\% \pm 3.51\%$ | **$83.11\% \pm 1.69\%$** |
| 100 | $88.40\% \pm 2.10\%$ | **$91.05\% \pm 0.85\%$** |
| **Full (Global)** | $97.50\%$ | **$98.39\%$** |

Structural Superiority
- Variance Reduction: The hybrid model reduces algorithmic variance by over 50% in low-data settings ($N \le 50$).
- Asymptotic Scaling: Even with unlimited data, the maximum-margin approach slightly outperforms the standard end-to-end baseline, reaching an upper bound of 98.39%.

The core finding represented in this table is that the Hybrid CNN-SVM architecture reduces algorithmic variance across random initializations by over 50% in highly constrained settings ($N \le 50$). This is achieved by replacing the probabilistic Softmax head—which overfits to local data density—with a deterministic Linear SVM that enforces a maximum geometric margin between class clusters.  

# 🔬 Implementation Details
- Frameworks: PyTorch for convolutional sculpting; Scikit-learn for geometric classification.  

- Training Protocol: The ResNet-18 backbone is initially trained via cross-entropy to "sculpt" the latent space before being frozen for SVM integration.  

