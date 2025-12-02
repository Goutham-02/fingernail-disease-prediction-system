# Nail Disease Diagnosis System

An AI-powered nail disease diagnosis system that combines **deep learning (CNN)**, **computer vision**, and **LLM reasoning** to provide accurate, transparent diagnoses of 17 different nail conditions.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)

---

## 🌟 Features

### Multi-Modal Diagnosis
- **CNN Model**: ResNet-based deep learning for visual pattern recognition (97%+ accuracy)
- **Computer Vision**: 14 feature detectors for clinical signs (lines, pitting, discoloration, etc.)
- **LLM Integration**: Gemini AI for symptom-based prediction refinement
- **Grad-CAM**: Visual explanations showing which nail regions influenced the prediction

### Transparency & Explainability
- **Separate Scores**: View Model, Visual, and Symptom scores independently
- **No Black Box**: All decision factors clearly displayed
- **Medical Context**: Detailed disease information with references

### API Versions
- **V1**: Basic CNN + CV fusion
- **V2**: Enhanced with LLM-based prediction refinement using user symptoms

---

## 📋 Supported Conditions

| Category | Diseases |
|----------|----------|
| **Autoimmune** | Alopecia Areata, Psoriasis, Darier's Disease |
| **Infections** | Onychomycosis (Fungal), Eczema |
| **Systemic Disease Indicators** | Terry's Nail, Muehrcke's Lines, Half-and-Half Nails, Clubbing |
| **Circulatory** | Splinter Hemorrhage, Bluish Nail, Pale Nail, Red Lunula |
| **Nutritional** | Koilonychia (Spoon Nails), Leukonychia |
| **Trauma/Structural** | Beau's Lines, Onycholysis, Yellow Nails, White Nail |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nail-disease-diagnosis.git
cd nail-disease-diagnosis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
# Required: Gemini API for LLM features
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Cloudinary for Grad-CAM image hosting
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### Running the Server

```bash
# V1 API (Basic)
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# V2 API (with LLM Refinement)
python -m uvicorn server_v2:app --host 0.0.0.0 --port 8001 --reload
```

Access the API:
- **Health Check**: `http://localhost:8000/health`
- **Web UI**: `http://localhost:8000/ui`
- **API Docs**: `http://localhost:8000/docs`

---

## 📡 API Usage

### Diagnose with Images

```bash
curl -X POST "http://localhost:8000/diagnose" \
  -F "images=@nail_photo.jpg" \
  -F "symptoms=yellow discoloration, thickening"
```

### Response Format (V2)

```json
{
  "cnn_prediction": {
    "predicted_class": "onychomycosis",
    "confidence": 0.72
  },
  "llm_refined_prediction": {
    "predicted_class": "onychomycosis",
    "confidence": 0.89,
    "reasoning": "The yellow discoloration and thickening strongly align with fungal infection...",
    "was_refined": true,
    "differential_concerns": ["psoriasis", "eczema"]
  },
  "final_prediction": "onychomycosis",
  "final_confidence": 0.89,
  "gradcam_urls": ["https://cloudinary.../heatmap.png"],
  "description": "Onychomycosis is a fungal infection...",
  "disease_info": { /* full medical context */ }
}
```

---

## 🏗️ Architecture

```
┌─────────────────┐
│  User Input     │
│  - Images (1-5) │
│  - Symptoms     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│       Integrated Diagnosis System       │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │   CNN    │  │    CV    │  │ LLM  │ │
│  │ ResNet   │  │ Features │  │Gemini│ │
│  │ 97.8%    │  │ 14 types │  │ 1.5  │ │
│  └────┬─────┘  └────┬─────┘  └──┬───┘ │
│       │             │            │     │
│       └──────┬──────┴────────────┘     │
│              │                         │
│        ┌─────▼─────┐                   │
│        │  Fusion   │                   │
│        │  Engine   │                   │
│        └─────┬─────┘                   │
└──────────────┼─────────────────────────┘
               │
         ┌─────▼─────┐
         │  Output   │
         │ + Grad-CAM│
         └───────────┘
```

### Key Components

- **`integrated_nail_diagnosis.py`**: Core diagnosis engine
- **`server.py`**: V1 FastAPI server
- **`server_v2.py`**: V2 with LLM refinement
- **`data1.json`**: Disease database (symptoms, features, treatments)
- **`17classes_resnet_97.pth`**: Pre-trained CNN model

---

## 🔬 Technical Details

### Visual Feature Detection

The system detects 14 programmatic features:

| Feature | Description | Diseases |
|---------|-------------|----------|
| `vertical_lines` | Vertical ridges/lines | Splinter Hemorrhage, Darier's |
| `horizontal_lines` | Beau's/Muehrcke's lines | Systemic illness indicators |
| `nail_pitting` | Small dents | Psoriasis, Alopecia Areata |
| `red_areas` | Hemorrhaging/redness | Splinter Hemorrhage |
| `yellow_discoloration` | Yellow nails | Fungal infection, Yellow Nail Syndrome |
| `white_discoloration` | Leukonychia | Various causes |
| `spoon_shape` | Koilonychia | Iron deficiency |
| `nail_clubbing` | Curved, bulbous nails | Lung/heart disease |
| And 6 more... | | |

### Grad-CAM Visualization

Enhanced Grad-CAM with:
- **HiResCAM-like** element-wise multiplication
- **Edge-guided refinement** for precise activation maps
- **Thresholding** to highlight only relevant regions

---

## 📊 Performance

- **Model Accuracy**: 97.8% on test set
- **Feature Detection**: OpenCV-based, calibrated on medical data
- **LLM Reasoning**: Gemini 1.5 Flash for symptom matching
- **Inference Time**: ~1-2 seconds per image (CPU), <500ms (GPU)

---

## 🛠️ Development

### Project Structure

```
nail-disease-diagnosis/
├── server.py                  # V1 API
├── server_v2.py              # V2 API (LLM)
├── integrated_nail_diagnosis.py  # Core engine
├── data1.json                # Disease database
├── 17classes_resnet_97.pth   # CNN weights
├── templates/                # Web UI
│   ├── index.html
│   ├── diagnose.html
│   └── health.html
├── static/                   # CSS/JS assets
├── output/                   # Diagnosis results
├── temp_uploads/            # Temporary image storage
└── requirements.txt
```

### Running Tests

```bash
python test_systems.py
```

### Example Usage

```python
from integrated_nail_diagnosis import NailDiagnosisSystem

# Initialize
system = NailDiagnosisSystem(
    model_path="17classes_resnet_97.pth",
    disease_data_path="data1.json"
)

# Diagnose
results = system.diagnose(
    image_paths=["nail1.jpg", "nail2.jpg"],
    symptoms="yellow nails, thickening"
)

print(results["aggregated_prediction"])
```

---

## 📝 API Endpoints

### V1 & V2 Common Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | System health check |
| `/diagnose` | POST | Main diagnosis endpoint |
| `/ui` | GET | Web interface |
| `/ui/diagnose` | GET | Diagnosis UI |
| `/ui/health` | GET | Health check UI |

---

## 🔒 Privacy & Security

- **No Data Storage**: Images deleted immediately after processing
- **HTTPS Ready**: Deploy with SSL certificates
- **API Key Protection**: Environment variables for sensitive data
- **Gradual CAM Upload**: Optional (can disable Cloudinary)

---

## ⚠️ Medical Disclaimer

**This system is for educational and research purposes only.**

- NOT a substitute for professional medical diagnosis
- Always consult a qualified dermatologist or physician
- AI predictions should be verified by medical professionals
- Not FDA approved or certified for clinical use

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ResNet Architecture**: He et al. (2015)
- **Grad-CAM**: Selvaraju et al. (2017)
- **Gemini AI**: Google DeepMind
- **Medical References**: DermNet NZ, Mayo Clinic, Cleveland Clinic

---

## 📧 Contact

For questions or collaboration:
- **Issues**: [GitHub Issues](https://github.com/yourusername/nail-disease-diagnosis/issues)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Multi-language support
- [ ] Expanded disease database (20+ conditions)
- [ ] Real-time video diagnosis
- [ ] Integration with EHR systems
- [ ] Federated learning for privacy-preserving model updates

---

**Built with ❤️ for better healthcare accessibility**
