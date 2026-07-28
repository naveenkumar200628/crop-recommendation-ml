# 🌾 Crop Recommendation System using Machine Learning

This project is a **machine learning–based crop recommendation system** that suggests the most suitable crop to grow based on soil and environmental conditions.  
It helps farmers and agricultural planners make **data-driven decisions** to improve productivity.

---

## 🚀 Features
- Predicts the best crop based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Humidity
  - Rainfall
- Trained using supervised machine learning
- Implemented in **Python** using **Jupyter Notebook**
- Easy to extend into a **web app (Flask / Streamlit)**

---

## 🧠 Machine Learning Workflow
1. Data Loading
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Model Training
6. Model Evaluation
7. Prediction

---

## 🛠️ Technologies Used
- Python 🐍
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📂 Project Structure
├── Crop_Model.ipynb # Main notebook (model building & training)
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore # Files ignored by Git

---
## 📊 Model Performance

### Confusion Matrix
![Confusion Matrix](images\confusion_matrix.png)

### Feature Importance
![Feature Importance](images\feature_importance.png)

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/crop-recommendation-system.git
cd crop-recommendation-system
python -m venv venv
#Create virtual environment (optional but recommended)
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
#Open Crop_Model.ipynb and run all cells.

📊 Model Output

The model predicts the most suitable crop based on input soil and weather conditions.
Example:
Input: N=90, P=42, K=43, Temperature=20.8, Humidity=82, pH=6.5, Rainfall=202
Output: Rice



👨‍💻 Author

Naveen Kumar
BCA – Data Science
VELS University