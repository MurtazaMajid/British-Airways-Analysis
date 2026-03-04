<div align="center">

# British Airways Booking Analysis and Prediction

### A data science project completed as part of the British Airways Virtual Experience Program on Forage

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Forage](https://img.shields.io/badge/British%20Airways-Forage%20Program-004499?style=for-the-badge)](https://www.theforage.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> Can we predict whether a customer will complete their flight booking before they do it? This project digs into real British Airways customer booking data, uncovers patterns in how and why people book flights, and builds machine learning models to predict booking completion, including handling a serious class imbalance in the data using SMOTE.

<br/>

**[Notebook](notebooks/) &nbsp;|&nbsp; [Visuals](images/)**

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Highlights](#project-highlights)
- [Data Description](#data-description)
- [Methodology](#methodology)
  - [Data Preprocessing and Feature Engineering](#1-data-preprocessing-and-feature-engineering)
  - [Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [Handling Class Imbalance with SMOTE](#3-handling-class-imbalance-with-smote)
  - [Model Training and Comparison](#4-model-training-and-comparison)
- [Results](#results)
- [Key Insights](#key-insights)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Author](#author)

---

## Project Overview

This project was completed as part of the **British Airways Data Science Virtual Experience Program** on Forage. The program simulates real work done by British Airways data science teams, giving a hands-on view into how airlines use data to understand customer behaviour.

The central question the project addresses is straightforward: given what we know about a customer at the time of booking, such as where they are booking from, what type of trip they are taking, which sales channel they used, and what add-ons they selected, can we predict whether they will actually complete their booking?

Two separate tasks were completed as part of this program:

**Task 1** covers analysis of the British Airways summer flight schedule, including passenger tier eligibility across different time-of-day categories, haul types, and arrival regions. A lookup table is generated from the grouped data.

**Task 2** is the main focus of this repository: exploratory data analysis and predictive modelling on customer booking data, including dealing with heavy class imbalance and comparing multiple machine learning models.

---

## Project Highlights

| Area | Detail |
|---|---|
| Exploratory Data Analysis | In-depth analysis of booking patterns, routes, preferences, and customer behaviour |
| Feature Engineering | Created an `Urgent_Booking` flag from `purchase_lead` to capture last-minute bookings |
| Class Imbalance Handling | Applied SMOTE to address an 88/12 class split between incomplete and complete bookings |
| Multiple Models Compared | Logistic Regression, Random Forest, and XGBoost trained and evaluated side by side |
| Balanced vs Unbalanced | Each model tested on both the raw imbalanced data and SMOTE-resampled data |
| Program Context | Completed as part of the British Airways Virtual Experience on Forage |

---

## Data Description

The dataset used in this project is the **British Airways customer booking dataset** provided exclusively through the British Airways Virtual Experience Program on Forage. Due to the proprietary nature of the data, it is not included in this repository and cannot be shared publicly.

To access the dataset and reproduce this analysis, you will need to enrol in the [British Airways Virtual Experience Program on Forage](https://www.theforage.com/simulations/british-airways/data-science-yqoz) and download the data directly from there. Enrolment is free.

Once you have the file, place `customer_booking.csv` in the root of the project directory and the notebook will run as expected.

**The dataset contains the following columns:**

| Column | What it contains |
|---|---|
| `num_passengers` | Number of passengers in the booking |
| `sales_channel` | How the booking was made (Internet or Mobile) |
| `trip_type` | Round Trip, One Way, or Circle Trip |
| `purchase_lead` | Days between the booking date and the travel date |
| `length_of_stay` | Number of nights at the destination |
| `flight_hour` | Hour of the day the flight departs |
| `flight_day` | Day of the week the flight departs |
| `route` | Origin to destination route code |
| `booking_origin` | Country where the booking was made |
| `wants_extra_baggage` | Whether the customer opted for extra baggage |
| `wants_preferred_seat` | Whether the customer chose a preferred seat |
| `wants_in_flight_meals` | Whether the customer added in-flight meals |
| `flight_duration` | Total flight duration in hours |
| `booking_complete` | Target variable: 1 if the booking was completed, 0 if not |

---

## Methodology

### 1. Data Preprocessing and Feature Engineering

The dataset arrives in fairly clean condition with no missing values, but a few steps are needed before modelling:

- `flight_day` is stored as text abbreviations (Mon, Tue, etc.) and is converted to integers 1 through 7 so it can be used numerically
- Numerical and categorical columns are separated for different encoding strategies
- Numerical features are scaled using `MinMaxScaler`
- Categorical features (`sales_channel`, `trip_type`, `booking_origin`) are encoded using `OneHotEncoder`
- The processed numerical and encoded categorical frames are concatenated into a single feature matrix for modelling

One new feature is also created:

**`Urgent_Booking`** is a binary flag set to 1 if the customer booked within 7 days of their travel date. The idea is that last-minute bookers may behave differently from those who plan well in advance, and this signal could help the model pick up on that difference.

---

### 2. Exploratory Data Analysis

Before any modelling, the data is explored thoroughly to understand who British Airways customers are and what their booking behaviour looks like. All analysis covers the full dataset as well as the subset of customers who actually completed their bookings.

**Booking Completion Status**

The most important thing to notice straight away is that only around 12% of bookings are completed. This heavy imbalance is the central challenge for modelling and shapes every decision made in the training process.

![Booking Status](images/booking%20status.png)

**Top Booking Countries**

Some countries generate significantly more bookings than others. The top 5 countries account for a disproportionately large share of all bookings, both completed and not completed.

![Most Bookings](images/most%20bookings.png)

![Most Bookings Completed vs Not Completed](images/most%20bookings%20completed%20and%20not%20completed.png)

**Sales Channel**

The majority of bookings come through the Internet rather than Mobile. This reflects the general trend in airline booking where web-based platforms still dominate over mobile apps.

![Sales Channel](images/sales%20channel.png)

**Trip Type**

Round trips are by far the most common booking type among customers, with one-way and circle trips making up a much smaller share.

![Trip Type](images/trip%20type.png)

**Most Common Routes**

Among customers who completed their bookings, a small number of routes account for a large share of completed reservations. The Auckland to Kuala Lumpur route (AKLKUL) is the single most common route in this group.

![Common Routes](images/common%20routes.png)

**Flight Departure Hours**

Completed bookings cluster around certain departure hours, suggesting that time of day plays a role in whether customers follow through with a reservation.

![Flight Hours](images/flight%20hours.png)

**Flight Duration**

Most completed bookings are for longer flights, suggesting that customers who go through with their bookings tend to be travelling greater distances.

![Flight Duration](images/flight%20duration.png)

**Number of Passengers**

The vast majority of completed bookings are for a single passenger, with small groups of two or three being the next most common.

![Number of Passengers](images/number%20pass.png)

**Extra Baggage**

A notable portion of customers who complete their bookings do not opt for extra baggage, though a meaningful minority do.

![Extra Baggage](images/extra%20baggage.png)

**Preferred Seat**

Most customers with completed bookings do not select a preferred seat, suggesting that seat selection is not a primary driver of booking completion.

![Preferred Seat](images/preferred%20seat.png)

**In-Flight Meals**

In-flight meal selection also skews toward not wanting meals among customers who completed bookings. The signal is kept as a feature since it may still carry some predictive value in combination with other columns.

![In-Flight Meals](images/in%20flight%20meals.png)

---

### 3. Handling Class Imbalance with SMOTE

The booking completion rate sits at roughly 12%, meaning about 88% of records are incomplete bookings. If a model is trained on this raw split, it will naturally learn to predict "not completed" most of the time and still report high accuracy. This would be a misleading result.

To fix this, **SMOTE (Synthetic Minority Oversampling Technique)** is applied. SMOTE generates new synthetic examples of the minority class by interpolating between existing minority class samples, rather than simply copying them. This produces a balanced training set where the model gets a fair chance to learn patterns from both outcomes.

Each of the three models is trained and evaluated twice: once on the original imbalanced data and once on the SMOTE-resampled data, so the effect of balancing can be directly compared.

---

### 4. Model Training and Comparison

Three models are trained and compared across both the imbalanced and SMOTE-balanced datasets.

**Logistic Regression** serves as the baseline. It is simple, fast, and easy to interpret, making it a useful reference point against which the tree-based models can be measured.

**Random Forest** is an ensemble of decision trees that handles non-linear relationships well. With 100 trees and a fixed random state, it typically outperforms logistic regression on tabular data with mixed feature types.

**XGBoost** builds trees one at a time, with each new tree correcting the mistakes of the previous ones. It is generally the strongest performer on structured tabular data and is the final recommended model here.

All models use an 80/20 train-test split. Evaluation metrics include accuracy, precision, recall, F1 score, and a confusion matrix heatmap for each configuration.
```python
# The five model configurations trained and evaluated:

# 1. Logistic Regression on raw imbalanced data
# 2. Random Forest on raw imbalanced data
# 3. XGBoost on raw imbalanced data
# 4. Random Forest on SMOTE-resampled data
# 5. XGBoost on SMOTE-resampled data
```

---

## Results

| Model | Data | Notes |
|---|---|---|
| Logistic Regression | Imbalanced | High accuracy but poor recall on completed bookings |
| Random Forest | Imbalanced | Similar issue, minority class largely missed |
| XGBoost | Imbalanced | Same pattern without resampling |
| Random Forest | SMOTE | Significantly better at identifying completed bookings |
| XGBoost | SMOTE | Best overall, strongest F1 on both classes |

The key takeaway is that raw accuracy is a misleading metric here. A model that always predicts "not completed" would still score around 88% accuracy. The real improvement from SMOTE shows up in the recall and F1 score for the completed-booking class, where models go from missing most of them to catching a meaningful proportion. XGBoost with SMOTE is the recommended model.

---

## Key Insights

Several patterns come out of the analysis that are worth highlighting from a business perspective.

The booking completion rate of around 12% is notably low and raises a question worth investigating: are customers dropping off due to pricing, a complicated checkout flow, or something else in the purchase process?

Customers booking from certain countries complete their bookings at higher rates than others. This could be useful for targeting specific markets with tailored campaigns or localised pricing.

The Auckland to Kuala Lumpur route has the highest volume of completed bookings among confirmed travellers, pointing to strong and reliable demand on this specific corridor.

Most completed bookings are for solo travellers on longer routes. This segment may respond well to targeted offers on comfort add-ons, since long-haul solo travellers tend to have more flexibility in what they spend.

Last-minute bookings, captured through the `Urgent_Booking` feature for customers booking within 7 days of travel, form a distinct segment that likely responds differently to pricing nudges compared to people planning weeks or months ahead.

---

## Repository Structure
```
British-Airways-Analysis/
|
+-- documents/                          Reports and task summaries
|
+-- images/
|   +-- booking status.png
|   +-- common routes.png
|   +-- extra baggage.png
|   +-- flight duration.png
|   +-- flight hours.png
|   +-- in flight meals.png
|   +-- most bookings completed and not completed.png
|   +-- most bookings.png
|   +-- number pass.png
|   +-- preferred seat.png
|   +-- sales channel.png
|   +-- trip type.png
|
+-- notebooks/
|   +-- Forage_British_Airways_.ipynb   Main analysis and modelling notebook
|
+-- README.md
```

---

## Quick Start

**Prerequisites:** Python 3.10+

**Clone the repository**
```bash
git clone https://github.com/MurtazaMajid/British-Airways-Analysis.git
cd British-Airways-Analysis
```

**Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn openpyxl jupyter
```

**Get the dataset**

Enrol in the [British Airways Virtual Experience Program on Forage](https://www.theforage.com/simulations/british-airways/data-science-yqoz) for free and download `customer_booking.csv`. Place it in the root of the project directory before running the notebook.

**Open the notebook**
```bash
jupyter notebook notebooks/Forage_British_Airways_.ipynb
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Analysis | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (MinMaxScaler, OneHotEncoder) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Models | Logistic Regression, Random Forest, XGBoost |
| Notebook | Jupyter |
| Program | British Airways Virtual Experience on Forage |

---

## Future Work

- Tune Random Forest and XGBoost hyperparameters using cross-validated grid search to push model performance further
- Test additional resampling strategies such as ADASYN or class weighting as alternatives to SMOTE
- Build a feature importance plot for the best-performing model to identify the top drivers of booking completion
- Explore whether `purchase_lead` alone is predictive enough to build a lightweight early-warning model
- Bring in external signals such as seasonality or route pricing to enrich the feature set

---

## Author

**Murtaza Majid**

[![GitHub](https://img.shields.io/badge/GitHub-MurtazaMajid-181717?style=flat-square&logo=github)](https://github.com/MurtazaMajid)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Murtaza%20Majid-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/murtaza-majid/)

---

<div align="center">

If you found this project useful, a star on the repository is appreciated.

Completed as part of the British Airways Virtual Experience Program on Forage.

</div>
