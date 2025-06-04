# Diamond Dataset analysis

A project for understanding, cleaning, analysing and predicting diamond prices based on various characteristics including carat weight, cut, color, and clarity in the dataset diamonds.csv.

## Description

This project analyzes a diamond dataset to create price prediction models. It includes:
- Data cleaning and preprocessing
- Exploratory data analysis
- Model training with different feature sets
- Web interface for price predictions using Streamlit

## Installation

1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository
```bash
git clone https://github.com/fendraq/AnalysingDiamondsDatasetPy.git
cd AnalysingDiamondsDatasetPy
```

3. Create and activate virtual environment using uv
```bash
uv venv .venv
source .venv/bin/activate  # Linux/WSL
```

4. Install dependencies with uv
```bash
uv pip install -r requirements.txt
```

5. Model files
Model files aren't included at this stage but run 

## Project Structure

```
├── data/                  # Data files
├── web_app/              # Streamlit web application
│   ├── pages/           # Application pages
│   └── models/          # Trained models
├── notebooks/            # Jupyter notebooks
│   ├── data_cleaning.ipynb
│   └── data_understanding.ipynb
└── requirements.txt      # Project dependencies
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run web_app/Home.py
```

2. Access the web interface at `http://localhost:8501`

3. Input diamond characteristics to get price predictions

## Models

The project uses two Random Forest models:
- Basic model using standard features
- Enhanced model incorporating carat bin features

Model files are stored separately due to size constraints. Use `download_models.py` to fetch required files.

## Features

- Interactive web interface
- Two prediction models
- Data visualization
- Comprehensive data analysis
- Feature importance analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - your.email@example.com
Project Link: [https://github.com/fendraq/AnalysingDiamondsDatasetPy](https://github.com/fendraq/AnalysingDiamondsDatasetPy)