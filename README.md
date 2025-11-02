# Mini Project - Bike Sharing Analysis

📘 Overview
This project analyzes the Bike Sharing Dataset to understand demand patterns and build predictive models for the count of rented bikes. The dataset records features such as weather conditions, season, humidity, temperature, and time-related factors, with the goal of exploring usage patterns and predicting total rentals. This detailed report documents every stage, from data acquisition to final model selection and interpretation of findings. The objective is not just prediction, but deep understanding of the factors driving bicycle rentals in an urban environment.

📂 Dataset Overview
The analysis utilizes data sourced from the publicly available UCI Bike Sharing Dataset, specifically focusing on the granular data provided in the daily (day.csv) and hourly (hour.csv) files. For this comprehensive report, we focus primarily on the hourly dataset to capture finer temporal dynamics.

Each record in the dataset represents a time slice (an hour or a day) and includes several crucial attributes:

### Time and Day Attributes (Categorical/Ordinal)
-season: Season of the year (1:spring, 2:summer, 3:fall, 4:winter).

-yr: Year (0: 2011, 1: 2012).

-mnth: Month (1 to 12).

-hr: Hour of the day (0 to 23).

-holiday: Whether the day is a holiday (Boolean).

-weekday: Day of the week (0:Sunday, 1:Monday, ..., 6:Saturday).

-workingday: Whether the day is a working day (Boolean).
