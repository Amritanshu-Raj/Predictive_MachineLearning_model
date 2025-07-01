# Medical Insurance Cost Prediction Project

## 📌 Overview
This project predicts individual medical insurance costs using machine learning. It analyzes how factors like age, BMI, smoking status, and children count affect insurance premiums. The solution includes:
- Data analysis notebook
- Machine learning model
- Interactive web application

![App Screenshot]([screenshot.png](https://github.com/Amritanshu-Raj/Predictive_MachineLearning_model/blob/main/Insurance_Charge_Predictionapp.jpeg?raw=true))

## 🔍 Key Findings from Analysis
Through detailed exploration of the insurance dataset, we discovered:

1. **Smoking is the biggest cost factor**  
   Smokers pay 238% more on average ($32,000 vs $9,400)

2. **BMI significantly impacts costs**  
   People with BMI > 30 pay 2x more than those with normal BMI

3. **Age matters more after 35**  
   Premiums increase sharply after age 35:
   ```
   18-35: $8,200 avg
   35-50: $12,400 avg
   50+: $16,100 avg
   ```

4. **Children have complex effects**  
   Costs decrease slightly for 1 child but increase for 3+ children

## ⚙️ Technical Implementation

### Data Processing Pipeline
```python
# Load and clean data
df = pd.read_csv('insurance.csv')
df = df.dropna()

# Remove outliers using Z-score
df = df[(np.abs(df["charges"]-df["charges"].mean())/df["charges"].std()<3)]

# Convert categorical features
df["smoker"] = df["smoker"].replace({"yes":1, "no":0})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Model Development
We tested multiple algorithms before selecting Gradient Boosting:

| Model | Training R² | Testing R² | Why Not Chosen |
|-------|-------------|------------|----------------|
| Linear Regression | 0.75 | 0.74 | Underfit data |
| Polynomial (deg=4) | 0.87 | 0.82 | Overfit data |
| **Gradient Boosting** | **0.95** | **0.84** | Best balance |

```python
final_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
final_model.fit(X_train, y_train)
```

## 🚀 Deployment
The Streamlit app provides simple interface for predictions:

```python
import streamlit as st

# User inputs
age = st.slider("Age", 18, 70, 30)
bmi = st.number_input("BMI", 18.0, 40.0, 25.0)
smoker = st.radio("Smoker", ["No", "Yes"])

# Convert and predict
smoker_val = 1 if smoker == "Yes" else 0
user_input = scaler.transform([[age, bmi, children, smoker_val]])
prediction = model.predict(user_input)

st.metric("Estimated Charge", f"${prediction[0]:,.2f}")
```

## 📂 Project Structure
```
insurance-project/
├── data/
│   └── insurance.csv           # Original dataset
├── models/
│   ├── gbr.pkl                 # Trained ML model
│   └── scaler.pkl              # Feature scaler
├── notebooks/
│   └── Insurance_Analysis.ipynb # Complete EDA & modeling
├── app/
│   └── Insurance_app.py        # Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```


## 🌟 Key Features
- **Interactive predictions**: Get instant cost estimates
- **Responsive design**: Works on mobile devices
- **Model transparency**: See key factors affecting your quote
- **Educational notebook**: Learn how the model was built

## 📈 Future Improvements
- [ ] Add regional pricing differences
- [ ] Include health condition factors
- [ ] Implement user accounts
- [ ] Create historical quote tracking

## 👨‍💻 About the Developer
As a fresher data analyst enthusiast, this project helped me learn:
- Practical data cleaning techniques
- Machine learning model selection
- Real-world deployment with Streamlit
- Communicating technical insights

Connect with me: ((https://www.linkedin.com/in/amritanshu-raj-104735233/))
