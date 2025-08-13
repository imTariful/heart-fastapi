## Heart Disease Prediction API 🚑❤️

A simple FastAPI application that predicts the presence of heart disease based on patient health data.
This project focuses on Dockerization and deployment rather than achieving high model accuracy.
Deployed on Render using a pre-trained Random Forest Classifier trained on the Heart Disease Dataset.

## 📁 Project Structure

```
Fastapi_ml/
├── app/
│   ├── main.py         # FastAPI app entry point
│   ├── schemas.py      # Pydantic model for request validation
├── model/
│   ├── heart_model.joblib  # Trained ML model
│   ├── meta.json           # Model metadata (features, model type)
├── train_model.py      # Script to train and save the model
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build configuration
├── docker-compose.yml  # Docker Compose configuration
└── README.md           # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
2. **Create and activate virtual environment**
   ```powershell
   # On Windows PowerShell
   python -m venv env
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\env\Scripts\activate
   
   # On Linux/Mac
   python -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already present)
   ```bash
   python model/train_model.py
   ```

5. **Run the API server**
   ```bash
   # For the main ML API
   uvicorn app.main:app --reload
   
   # Or for the async/sync demo
   uvicorn app.async_demo:app --reload
   ```

6. **Access the API**
   - API: http://127.0.0.1:8000
   - Interactive docs: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc

## 📖 API Endpoints

### Main ML API (`app.main`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check endpoint |
| GET | `/info` | Model information |
| POST | `/predict` | Predict Iris species |

### Async Demo API (`app.async_demo`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sync` | Synchronous operation (5s delay) |
| GET | `/async` | Asynchronous operation (5s delay) |

## 🔬 Usage Examples
```bash
# Using curl
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 63,
       "sex": 1,
       "cp": 3,
       "trestbps": 145,
       "chol": 233,
       "fbs": 1,
       "restecg": 0,
       "thalach": 150,
       "exang": 0,
       "oldpeak": 2.3,
       "slope": 0,
       "ca": 0,
       "thal": 1
     }'
```

```python
# Using Python requests
import requests
data = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
# Example output: {"heart_disease": true, "probability": 0.87}
```

### Load Testing

Run the included load test to compare async vs sync performance:

```bash
python load_test.py
```

This will send 5 concurrent requests to both endpoints and show the performance difference.

## 🧠 Model Details

**Algorithm:** Random Forest Classifier

**Dataset:** Heart Disease Dataset (Kaggle)

**Features:**

age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

**Target:** 0 (No Heart Disease), 1 (Heart Disease)

**Probability:** Optional output from predict_proba (if supported)

## ⚡ Performance Insights

This project demonstrates the key differences between synchronous and asynchronous operations:

- **Sync Endpoints**: Block the entire event loop during I/O operations
- **Async Endpoints**: Allow other requests to be processed concurrently
- **Practical Impact**: Async endpoints can handle multiple requests simultaneously, while sync endpoints process them sequentially

## 🛠️ Development

### Adding New Features

1. **New ML Models**: Add training scripts in `model/` directory
2. **New Endpoints**: Add routes in `app/main.py`
3. **Schema Changes**: Update `app/schemas.py` for request/response models

### Running Tests

```bash
# Load testing
python load_test.py

# Model training
python model/train_model.py
```

## 📦 Dependencies

- **fastapi**: Modern, fast web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI applications
- **scikit-learn**: Machine learning library for model training
- **joblib**: Efficient serialization for ML models
- **numpy**: Numerical computing library
- **pydantic**: Data validation using Python type hints
- **httpx**: Async HTTP client for testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

**Author :** [imTariful](https://github.com/imTariful)