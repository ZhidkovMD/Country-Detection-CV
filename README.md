# Solving Country Detection problem using dataset of synthetic passport images

> **Computer Vision model to detect country of origin from passport/ID images**  
> Built with PyTorch + ConvNeXt • 100% Validation Accuracy • <100ms Inference • Anti-Overfitting Design

---

The goal is to accurately identify the country of origin for each passport image

The model is based on **ConvNeXt-Tiny**, fine-tuned via **transfer learning** on a synthetic dataset of passport images from 24 countries.

I utilize PyTorch framework for model development

Achieved model inference time is less than 1 second per image

Classification model was designed to generalize beyond passports and work with a wide range of document types, including those without MRZ zones (e.g., ID cards, driver licenses, voter's cards). The solution is scalable to support hundreds of countries and varying document layouts, languages, and formats

I provide a well-documented Jupyter notebook with comments and explanations for each step of the process and a Python script for inference along with all necessary instructions for running and reproducing the results

Model weights and checkpoints are attached as well, you can find them in the "models" folder

Designed originally as an interview assignment for CV Engineer role - demonstrates full pipeline: data loading, model training, evaluation, and deployment-ready inference.

---

## Repository Structure
.
├── country_detection.ipynb # Full Jupyter Notebook (Colab-ready) — training + inference
├── inference_pipeline.py # Standalone script for single-image inference
├── models/ # Pre-trained model weights
│ └── best_country_model.pth
│ └── country_detection_model_checkpoint.pth
├── demo/ # Example image for quick demo
│ └── us.jpeg
├── requirements.txt # Python dependencies
├── .gitignore # Ignores cache, logs, datasets
└── README.md # This file

---

## How to Use

### Option 1: Run in Google Colab (Recommended)
1. Open `country_detection.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run all cells — it will download dataset, train model, and run inference on demo image.

### Option 2: Run Locally
```bash
# Clone repo
git clone https://github.com/your-username/country-detection-cv.git
cd country-detection-cv

# Install dependencies
pip install -r requirements.txt

# Run inference on demo image
python inference_pipeline.py demo/us.jpeg

---

## Disclaimer

This project is for **educational and demonstration purposes only**.  
The model was trained on **synthetic passport images** and is not intended for use in real-world document verification, identity validation, or any legally binding applications.

Any real passport image used in inference (e.g., in the `demo/` folder) is used **strictly for technical demonstration**.  
I do not claim ownership of any such image, nor do I store, distribute, or use it for any purpose beyond running this demo.

**I am not responsible** for:
- Any misuse of this software
- Any privacy violations or legal consequences arising from using real document images
- Any incorrect predictions or decisions based on model output

By using this project, you agree to:
- Use only synthetic or legally obtained images with proper consent
- Not use this software in production, legal, governmental, or security contexts without professional validation
- Assume all risks associated with inference on real personal documents

Respect privacy. Respect the law. Use responsibly.