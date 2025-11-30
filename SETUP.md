# Setup and Installation Guide

## Prerequisites

- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- pip package manager
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MukeshKumawat0903/End-to-End_Customer_Behavior_Analysis_and_Prediction_Platform.git
cd End-to-End_Customer_Behavior_Analysis_and_Prediction_Platform
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Application

Edit `config.yaml` to set your data paths and preferences:

```yaml
data:
  raw_data_path: "path/to/your/data.parquet"
  sample_size: 100000  # Set to null for full dataset
```

### 5. Run the Application

#### Option A: Streamlit Dashboard

```bash
streamlit run app/streamlit/dashboard.py
```

Access at: http://localhost:8501

#### Option B: FastAPI Server

```bash
python api.py
```

Access API docs at: http://localhost:8000/docs

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

## Project Structure

```
project_root/
├── api.py                      # FastAPI application
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── app/
│   └── streamlit/
│       └── dashboard.py        # Streamlit dashboard
├── src/
│   ├── config.py              # Configuration module
│   ├── schemas.py             # Data validation schemas
│   ├── data/
│   │   ├── load.py           # Data loading
│   │   └── preprocess.py     # Preprocessing pipeline
│   ├── models/
│   │   ├── train.py          # Model training
│   │   ├── evaluate.py       # Model evaluation
│   │   └── utils.py          # Model utilities
│   ├── visualization/
│   │   └── visualize.py      # Visualization utilities
│   └── utils/
│       ├── logger.py         # Logging utilities
│       └── mlflow_utils.py   # MLflow integration
├── tests/                      # Unit tests
└── .github/
    └── workflows/
        └── ci-cd.yml          # CI/CD pipeline
```

## Configuration Options

### Data Configuration

- `raw_data_path`: Path to raw data file
- `sample_size`: Number of rows to sample (null for full dataset)
- `test_size`: Test set proportion (0.0-1.0)
- `validation_size`: Validation set proportion (0.0-1.0)

### Model Configuration

- `type`: Model type (xgboost, lightgbm)
- `device`: Computing device (cpu, cuda)
- `hyperparameters`: Grid search parameters
- `search_method`: Search strategy (grid, random)

### MLflow Configuration

- `enabled`: Enable/disable MLflow tracking
- `tracking_uri`: MLflow tracking server URI
- `experiment_name`: Experiment name

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 100.0,
    "hour": 14,
    "is_weekend": false,
    "total_clicks": 10,
    "avg_session_time": 120.5,
    "purchase_freq": 0.3,
    "products_viewed": 5,
    "price_range": 50.0,
    "session_duration": 300.0,
    "recency": 10,
    "frequency": 3,
    "monetary": 500.0,
    "main_category": "electronics",
    "rfm_segment": 2
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "price": 100.0,
        "hour": 14,
        ...
      },
      {
        "price": 200.0,
        "hour": 15,
        ...
      }
    ]
  }'
```

## Development

### Running with Hot Reload

```bash
# API with auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Dashboard with auto-reload (enabled by default)
streamlit run app/streamlit/dashboard.py
```

### Code Quality

```bash
# Format code
black src/ tests/ api.py

# Lint code
flake8 src/ tests/ api.py

# Type checking
mypy src/
```

## MLflow Tracking

View experiment tracking:

```bash
mlflow ui --backend-store-uri mlruns
```

Access at: http://localhost:5000

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the project root and virtual environment is activated
2. **Config not found**: Check that `config.yaml` exists in project root
3. **Data path errors**: Update `data.raw_data_path` in config.yaml
4. **Memory issues**: Reduce `data.sample_size` in config.yaml

### Logging

Logs are saved to `logs/app.log`. Check this file for detailed error messages.

```bash
# View recent logs
tail -f logs/app.log
```

## Performance Optimization

### For Large Datasets

1. Increase sample size gradually
2. Use `device: cuda` in config for GPU acceleration (requires CUDA setup)
3. Adjust `model.cv_folds` to reduce cross-validation time

### For Production

1. Set `logging.level: WARNING` in config
2. Disable MLflow if not needed: `mlflow.enabled: false`
3. Use `search_method: random` with lower `n_iter` for faster tuning

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

See [LICENSE](LICENSE) file for details.
