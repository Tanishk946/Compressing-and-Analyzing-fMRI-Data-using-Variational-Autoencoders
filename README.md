# Compressing and Analyzing fMRI Data using Variational Autoencoders

## Project Overview
This project implements a Variational Autoencoder (VAE) to compress and analyze functional MRI (fMRI) data from the Human Connectome Project (HCP). The VAE compresses high-dimensional fMRI timeseries data into a lower-dimensional latent space, enabling efficient storage and analysis. The compressed representations are used for downstream tasks, such as classifying cognitive states or detecting anomalies in brain activity. The project also incorporates uncertainty quantification to evaluate model reliability, making it relevant for applications in scientific machine learning and neuroimaging.

This project aligns with the research interests of the Machine Intelligence Group and the Data Science & Analytics Group at Lawrence Livermore National Laboratory (LLNL), focusing on representation learning, data compression, uncertainty quantification, and scientific data analysis.

## Dataset
The project uses the Human Connectome Project–Young Adult (HCP-YA) dataset, available at [https://library.ucsd.edu/dc/object/bb59818382](https://library.ucsd.edu/dc/object/bb59818382). The dataset includes:
- **Data Type**: Preprocessed fMRI timeseries data and functional connectivity matrices from 1,200 healthy young adults.
- **Format**: NumPy arrays, suitable for machine learning tasks.
- **Applications**: Compression, cognitive state classification, and anomaly detection.

## Project Objectives
1. **Data Compression**: Compress fMRI timeseries data using a VAE to reduce storage requirements while preserving essential information.
2. **Reconstruction Evaluation**: Reconstruct the original data from latent representations and evaluate quality using metrics like mean squared error.
3. **Downstream Task**: Use latent representations for tasks like cognitive state classification or anomaly detection.
4. **Uncertainty Quantification**: Analyze latent space variance to quantify prediction uncertainty.

## Installation
### Prerequisites
- Python 3.8+
- Libraries: `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`
- Access to the HCP-YA dataset (download from [https://library.ucsd.edu/dc/object/bb59818382](https://library.ucsd.edu/dc/object/bb59818382))

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fmri-vae-compression.git
   cd fmri-vae-compression
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn
   ```
3. Download and preprocess the HCP-YA dataset, placing it in the project directory.

## Usage
1. **Preprocess Data**:
   - Load the fMRI dataset and normalize the timeseries data.
   - Update the data path in `vae_fMRI_compression.py` to point to your dataset.

2. **Run the VAE**:
   ```bash
   python vae_fMRI_compression.py
   ```
   - The script trains the VAE, reconstructs the data, and generates visualizations of original vs. reconstructed timeseries.

3. **Downstream Tasks**:
   - Modify the script to use latent representations for classification or anomaly detection (e.g., using scikit-learn classifiers or Isolation Forest).

4. **Visualizations**:
   - The script generates plots comparing original and reconstructed fMRI data.
   - Additional visualizations (e.g., latent space t-SNE) can be added for interpretability.

## Code Structure
- `vae_fMRI_compression.py`: Main script implementing the VAE, data preprocessing, training, and visualization.
- `data/`: Directory for storing the HCP-YA dataset (not included in the repository due to size).
- `outputs/`: Directory for saving model outputs and visualizations.

## Results
- **Compression**: The VAE achieves significant data reduction (e.g., from thousands of features to a 32-dimensional latent space) with low reconstruction error.
- **Downstream Performance**: Latent representations enable accurate classification of cognitive states (e.g., resting vs. task states) or detection of anomalous brain activity.
- **Uncertainty Quantification**: Variance analysis in the latent space provides insights into prediction reliability.

## Future Improvements
- Incorporate convolutional layers for spatial-temporal fMRI data.
- Apply graph neural networks to model functional connectivity matrices.
- Use SHAP or LIME for enhanced model interpretability.

## References
- Human Connectome Project: [https://library.ucsd.edu/dc/object/bb59818382](https://library.ucsd.edu/dc/object/bb59818382)
- TensorFlow VAE Tutorial: [https://www.tensorflow.org/tutorials/generative/cvae](https://www.tensorflow.org/tutorials/generative/cvae)
- fMRI Autoencoder Research: [https://www.sciencedirect.com/science/article/pii/S1053811921006984](https://www.sciencedirect.com/science/article/pii/S1053811921006984)

## Contact
For questions or contributions, contact [your-email@example.com](mailto:your-email@example.com).
