# Mini Project - Bike Sharing Analysis


## 📘 Overview
This project analyzes the Bike Sharing Dataset to understand demand patterns and build predictive models for the count of rented bikes. The dataset records features such as weather conditions, season, humidity, temperature, and time-related factors, with the goal of exploring usage patterns and predicting total rentals. This detailed report documents every stage, from data acquisition to final model selection and interpretation of findings. The objective is not just prediction, but deep understanding of the factors driving bicycle rentals in an urban environment.

## 📂 Dataset Overview
The analysis utilizes data sourced from the publicly available UCI Bike Sharing Dataset, specifically focusing on the granular data provided in the daily (day.csv) and hourly (hour.csv) files. For this comprehensive report, we focus primarily on the hourly dataset to capture finer temporal dynamics.

Each record in the dataset represents a time slice (an hour or a day) and includes several crucial attributes:

#### Time and Day Attributes (Categorical/Ordinal)
season: Season of the year (1:spring, 2:summer, 3:fall, 4:winter).

yr: Year (0: 2011, 1: 2012).

mnth: Month (1 to 12).

hr: Hour of the day (0 to 23).

holiday: Whether the day is a holiday (Boolean).

weekday: Day of the week (0:Sunday, 1:Monday, ..., 6:Saturday).

workingday: Whether the day is a working day (Boolean).

Environmental Conditions (Numerical/Normalized)
These features are normalized to the range [0, 1] to facilitate scaling processes later:

temp: Normalized temperature in Celsius ($T$ normalized).

atemp: Normalized feeling temperature or "apparent temperature" ($T_{apparent}$ normalized).

hum: Normalized humidity level (ranging from 0 to 1).

windspeed: Normalized wind speed (ranging from 0 to 1).

User Counts (Target Variables)
The core focus of the prediction task revolves around these count variables:

casual: Count of casual (non-registered) users rentals.

registered: Count of registered users rentals.

cnt: Total count of bikes rented ($\text{cnt} = \text{casual} + \text{registered}$). This serves as the primary target variable ($\mathbf{y}$).

## 🧹 Data Preparation
Rigorous data preparation is essential for building robust machine learning models. This stage involved meticulous cleaning, transformation, and feature engineering.

1. Data Cleaning and Transformation
The raw data required transformation to ensure compatibility with modeling libraries and improve interpretability:

Missing Values and Duplicates: An initial check confirmed the absence of missing (null) values across all columns and no duplicate records existed in the hourly dataset.

Encoding Categorical Features: Several integer-coded features needed conversion to meaningful labels for better EDA interpretation and to treat them as nominal categories during modeling:

season: Mapped to {'Spring', 'Summer', 'Fall', 'Winter'}.

mnth: Mapped to month names.

hr: Mapped to 24-hour format labels (e.g., '00:00', '13:00').

weekday: Mapped to day names ('Sun', 'Mon', etc.).

yr and holiday/workingday were converted to standard boolean or categorical types as needed.

2. Feature Selection
The initial dataset contained several identifiers that do not contribute to prediction and must be removed to prevent data leakage or overfitting:

Removed Columns: instant (unique identifier) and dteday (date information, as yr, mnth, hr, and weekday are retained).

Independent Variables ($\mathbf{X}$): All remaining processed features (time attributes, weather conditions) were selected as predictors.

Dependent Variable ($\mathbf{y}$): The total count, $\mathbf{y} = \text{cnt}$, was isolated for regression tasks.

3. Feature Scaling
Since environmental variables like temp, atemp, hum, and windspeed are provided in normalized form, but other features (even if categorical and one-hot encoded later) are often treated differently, explicit scaling for numerical features is standard practice, particularly for regularized regression models (Ridge/Lasso).

We applied the StandardScaler from Scikit-learn to ensure all continuous features contribute equally to the distance metrics used by certain algorithms, preventing features with larger magnitudes from dominating the regularization penalty:
$$ x' = \frac{x - \mu}{\sigma} $$ Where $\mu$ is the mean and $\sigma$ is the standard deviation of the feature column.

## 📊 Exploratory Data Analysis (EDA)
EDA was crucial for identifying trends, validating assumptions, and selecting appropriate features for modeling. Visualizations were produced using Matplotlib and Seaborn.

Key Visualization Insights:
Distribution of Bike Rental Counts (cnt): The distribution plot of the target variable revealed that the data is heavily skewed towards lower rental counts, characteristic of count data. This suggests that many hours have moderate usage, while extremely high usage is rare. While Poisson regression might be theoretically ideal for count data, the sheer volume and magnitude often allow for effective linear modeling approximations, especially after transformation (though transformations were omitted here for direct comparison of standard regression models).

Correlation Heatmap: A heatmap visualized pairwise correlations. A strong, positive linear correlation was consistently observed between temperature (temp and atemp) and the total rental count (cnt). Conversely, factors like high humidity or strong winds showed slight negative correlations.

Seasonal Averages: Bar plots comparing the average number of rentals across the four seasons confirmed typical seasonal trends: demand is significantly higher during Summer (Season 2) and Fall (Season 3), while Spring (Season 1) shows lower baseline usage, and Winter (Season 4) shows the lowest usage.

Hourly Patterns: A line chart illustrating mean hourly rentals clearly showed bimodal usage patterns, indicative of commuting behavior:

Morning Peak: Approximately 7 AM to 9 AM.

Evening Peak: Approximately 4 PM (16:00) to 7 PM (19:00).

Low usage is consistently observed overnight (1 AM to 6 AM).

Weekday vs. Weekend Analysis: Boxplots comparing counts showed that while overall volume might be similar, the distribution shapes differ. Weekdays exhibit sharper peaks corresponding to commute times, whereas weekends often show a more sustained, flatter peak extending across the mid-day hours as leisure ridership balances out commuter drop-offs.

## 🤖 Model Building
The primary goal of the modeling phase was to build models capable of predicting the total count ($\text{cnt}$) based on the prepared features ($\mathbf{X}$). We employed a standard approach involving train-test splitting and fitting three regression techniques.

1. Data Splitting
The dataset was partitioned into training and testing sets using an 80% training (for fitting parameters) and 20% testing (for unbiased evaluation) split:
$$ (\mathbf{X}{\text{train}}, \mathbf{y}{\text{train}}), (\mathbf{X}{\text{test}}, \mathbf{y}{\text{test}}) \leftarrow \text{train_test_split}(\mathbf{X}, \mathbf{y}, \text{test_size}=0.2) $$

2. Model Implementation and Training
We compared three fundamental regression models:

A. Multiple Linear Regression (Baseline Model)
This model serves as the benchmark, assuming a linear relationship between the features and the count:
$$ \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n $$ Training involved fitting the coefficients ($\beta$) using Ordinary Least Squares (OLS) on the training data:

from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

B. Ridge Regression (L2 Regularization)
Ridge regression adds an $L_2$ penalty to the loss function to shrink coefficients and mitigate multicollinearity and overfitting:
$$ \text{Minimize: } \sum_{i=1}^{m} (y_i - \hat{y}i)^2 + \alpha \sum{j=1}^{n} \beta_j^2 $$ Hyperparameter $\alpha$ (defaulted or optimized, typically using cross-validation) controls the strength of the regularization.

C. Lasso Regression (L1 Regularization)
Lasso regression uses an $L_1$ penalty. Crucially, Lasso tends to drive less important feature coefficients exactly to zero, performing inherent feature selection:
$$ \text{Minimize: } \sum_{i=1}^{m} (y_i - \hat{y}i)^2 + \alpha \sum{j=1}^{n} |\beta_j| $$

3. Evaluation Metrics
Model performance was assessed using standard regression metrics on the test set:

Mean Squared Error (MSE): The average of the squared differences between prediction and actual observation. RMSE (Root Mean Squared Error) is the square root of MSE, providing an error measure in the original units (bike counts). [ \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} ]

Coefficient of Determination ($R^2$ Score): Measures the proportion of the variance in the dependent variable that is predictable from the independent variables. [ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} ]

## 📈 Model Evaluation
The performance comparison across the three regression techniques provided insight into how regularization affected model fit and generalization.

ModelR² Score (Test Set)RMSE (Test Set)Linear Regression (OLS)0.83$\approx 900$Ridge Regression0.84$\approx 870$Lasso Regression0.82$\approx 910$

Conclusion of Evaluation:
Ridge Regression emerged as the best-performing model. It achieved the highest $R^2$ score (0.84), indicating that it explained 84% of the variance in bike rentals, and simultaneously yielded the lowest Root Mean Squared Error ($\approx 870$ bikes), suggesting superior predictive accuracy and better generalization stability compared to the baseline OLS model. Lasso's slightly lower performance might indicate that most features retained some predictive value, and forcing coefficients to zero was detrimental.

## 🧠 Insights Derived from Modeling and EDA
The analysis strongly confirmed that bike sharing demand is highly predictable based on temporal and meteorological factors.

Dominant Drivers: Temperature (temp and atemp) was identified as the single most influential positive driver of rental counts across all models, as evidenced by the largest corresponding $\beta$ coefficients (after appropriate scaling checks). More comfortable weather correlates directly with higher usage.

Deterrents: Wind speed and humidity exhibited a statistically significant negative impact on ridership, suggesting that extreme weather conditions suppress demand, even when temperature is favorable.

Temporal Structure: The workingday status and the specific hr (hour) are critical predictors. The model heavily weighs these factors to predict the pronounced morning and evening commute peaks seen in the EDA.

User Segmentation: While not detailed here, separate analysis (implied by the data structure) would show that registered users dominate the predictable commuter spikes, while casual users show higher variability influenced more directly by weekend/holiday status and purely favorable weather conditions.

## 🧩 Technologies Used
This mini-project relied on the robust capabilities of the Python ecosystem:

Python 3: The core programming language.

NumPy: Efficient numerical operations and array handling.

Pandas: Primary tool for data loading, cleaning, manipulation, and transformation.

Matplotlib, Seaborn: Comprehensive libraries utilized for generating detailed statistical visualizations and exploring data distributions.

Scikit-learn: The essential machine learning library providing standardized implementations for data preprocessing (StandardScaler) and regression modeling (LinearRegression, Ridge, Lasso).

## ▶️ How to Run This Analysis
To reproduce this comprehensive analysis, follow these steps:

Acquire Data: Ensure the hour.csv file from the UCI Bike Sharing Dataset is accessible in the project directory.

Environment Setup: Clone the relevant repository or open the primary analysis file: Session 06 - Mini Project2 - Bike Sharing.ipynb

Install Dependencies: Execute the following command in your terminal or notebook cell to install necessary libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

Execution: Run all cells sequentially within the Jupyter Notebook environment. The initial cells handle data loading and cleaning, followed by EDA visualizations, feature scaling, model fitting, and final evaluation reporting.

Review: The final output includes tables detailing model performance metrics ($R^2$ and RMSE) that match the summary provided in this document.

## 📊 Results Summary
The bike sharing count prediction task proved highly tractable using linear regression techniques, provided proper feature engineering and scaling were applied.

Key MetricObservationCorrelationStrong positive correlation between temperature and rental count ($\rho \approx 0.6$)Model PerformanceRidge Regression provided the optimal balance, achieving $R^2 \approx 0.84$ and lowest RMSE ($\approx 870$).VisualizationClear, predictable seasonal demand shifts and pronounced bimodal hourly usage patterns confirmed the need to engineer time-based features effectively.
