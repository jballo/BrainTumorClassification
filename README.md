# Brain Tumor Classification

This project leverages deep learning for classifying brain tumor MRI scans into four categories: glioma, meningioma, no tumor, and pituitary. The implementation includes both a transfer learning model based on the Xception architecture and a custom CNN model that can be trained, evaluated, and utilized within an interactive Streamlit web app.

## Overview

- **Data Preparation**:  
  The project uses a publicly available MRI dataset (downloaded via Kaggle) to create training, validation, and testing splits. A helper function parses the directory structure to create data frames with image paths and corresponding classes.

- **Model Architecture**:  
  Two modeling strategies are implemented:
  - **Transfer Learning with Xception**: A pre-trained Xception model is fine-tuned (using additional dense layers, dropout, and flattening) for tumor classification.
  - **Custom CNN**: A convolutional neural network designed from scratch with multiple convolutional, pooling, dropout, and fully connected layers.

- **Visualization**:  
  The notebook includes routines to generate plots for class distributions, training/validation metrics, and confusion matrices. A saliency map is also computed to highlight the regions of interest in the MRI scans.

- **Interactive Web App**:  
  A Streamlit-based web interface enables users to upload an MRI image and receive a classification along with an explanation generated via multimodal LLM integration. The app supports various LLM endpoints (e.g., Gemini-1.5-Flash, OpenAI 4.0-Mini) for generating contextual explanations based on the saliency map.

## Requirements

- Python 3.8+
- TensorFlow and Keras
- TensorFlow Hub (for Xception weights)
- scikit-learn, pandas, numpy, matplotlib, seaborn
- OpenCV
- Streamlit, pyngrok, python-dotenv
- Additional dependencies as specified in the notebook cells

## Setup & Installation

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd BrainTumorClassification
   ```

2. **Install Dependencies**  
   It is recommended to set up a virtual environment. Then run:
   ```bash
   pip install -r requirements.txt
   ```
   If there isn’t a `requirements.txt`, refer to the dependencies listed in the notebook cells and install them manually.

3. **Dataset**  
   The dataset is downloaded directly via Kaggle.  
   Ensure you have Kaggle properly configured:
   ```bash
   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip
   ```

## Running the Notebook

Launch the Jupyter Notebook or JupyterLab and open `BrainTumorClassification.ipynb`. The notebook is structured into cells that:
- Load the data and perform preprocessing.
- Train both the transfer learning and custom CNN models.
- Evaluate performance using metrics, confusion matrices, and visualizations.
- Generate and display saliency maps to aid in interpretability.

## Using the Streamlit Web App

The project includes an interactive web app implemented with Streamlit. To run the app:

1. Ensure you have set up the environment variables in a `.env` file (specifically the `GOOGLE_API_KEY` and `OPENAI_API_KEY` if using LLM capabilities).
2. Start the Streamlit server by running:
   ```bash
   streamlit run app.py
   ```
3. Optionally, if running on platforms like Colab, the code sets up an ngrok tunnel to expose the web app externally.

## License

This project is licensed under the [MIT License](LICENSE).

## Final Remarks

This solution integrates state-of-the-art deep learning methods with robust model evaluation and an interactive user interface to provide transparency in predictions. The saliency maps provide insight into model decision-making, while the use of multimodal LLMs enables dynamic, on-demand explanations that can be invaluable in clinical or research contexts.