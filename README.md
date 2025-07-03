### 🔁 Customer Churn Prediction ML Pipeline

This project provides a **production-ready, end-to-end machine learning pipeline** for predicting customer churn using classification algorithms and modern data science tools. It includes **data preprocessing**, **EDA**, **model training**, **hyperparameter tuning**, **evaluation**, and **deployment via Flask API**.

---

## 📂 Project Structure

```

Customer-Churn-Prediction-with-Hyperparameter-Optimization-and-Model-Deployment/
│
├── data/                          # Raw dataset
│   └── churn_data.csv
│
├── src/                           # Core Python scripts
│   ├── preprocessing.py           # Data loading and preprocessing
│   ├── eda_visualization.py       # Data visualization functions
│   ├── model_training.py          # Base ML training script
│   ├── hyperparameter_tuning.py   # GridSearchCV optimization
│   ├── model_evaluation.py        # Evaluation metrics and reports
│   └── utils.py                   # Optional helper functions
│
├── app/
│   └── app.py                     # Flask REST API for predictions
│
├── notebooks/
│   └── churn_eda.ipynb            # Jupyter Notebook for EDA
│
├── models/
│   └── churn_model.pkl            # Trained model saved with joblib
│
├── requirements.txt               # Python dependencies
└── README.md                      # You're here

````

---

## 📌 Objective

To develop a machine learning pipeline capable of predicting customer churn with high accuracy. The pipeline supports:
- Feature engineering
- Visualization
- Model selection
- Hyperparameter tuning
- API deployment for real-world inference

---

## 💼 Use Case

**Industry Example:** Telecom or subscription-based services

**Business Value:** Helps reduce churn by identifying at-risk customers and enabling retention strategies like offers, feedback, and targeted communication.

---

## 🛠️ Tech Stack

| Category               | Tools Used                          |
|------------------------|-------------------------------------|
| Programming Language   | Python                              |
| Data Manipulation      | pandas, numpy                       |
| Visualization          | seaborn, matplotlib                 |
| ML Algorithms          | scikit-learn, XGBoost               |
| Hyperparameter Tuning  | GridSearchCV                        |
| Model Serialization    | joblib                              |
| Deployment             | Flask                               |
| Notebook Environment   | Jupyter Notebook                    |

---

## 📈 Model Training

Currently uses **Random Forest** and **XGBoost** as base classifiers. The training script can be extended to include other models.

📂 `src/model_training.py` trains the model and saves it to `models/churn_model.pkl`.

---

## 🧪 Example Prediction API

Run the API:
```bash
cd app/
python app.py
````

Test using `curl` or Postman:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
-d '{"features": [0.0, 1.0, 45.0, 5000.0, 60.0, 1.0, 0.0]}'
```

Response:

```json
{
  "churn_prediction": 1
}
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/churn_ml_pipeline.git
cd churn_ml_pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run EDA Notebook

```bash
jupyter notebook notebooks/churn_eda.ipynb
```

---

## 📊 Dataset

You can use the **Telco Customer Churn dataset** from Kaggle or IBM:

🔗 [Download Here (GitHub)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

Save as:

```bash
data/churn_data.csv
```

---

## 🧠 Model Insights

* Handles numerical + categorical data
* Supports hyperparameter tuning
* Scalable for more complex models (e.g., neural networks)
* Modular structure for experimentation

---

## 👨‍💻 Author

**Harsh Sonkar**
Machine Learning Engineer | Data Scientist
[LinkedIn](https://www.linkedin.com/in/harsh-sonkar/) | [GitHub](https://github.com/harsh-sonkar)

---

## 🤝 Contributions

Pull requests are welcome! Please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is licensed under the MIT License.

```

---

Would you like me to:
- Generate a zipped version of the project with all these files?
- Deploy this to Streamlit or HuggingFace Spaces?
- Add deep learning (Keras/TensorFlow) into the pipeline?

Let me know your next step!
```
