# Data-Masters-Challenge-Team-3

This repository contains the code and reports for Team 3’s submission to the Data Masters Challenge Olympiad, aimed at achieving a 20% energy waste reduction across multiple buildings. The project includes data cleaning, exploratory data analysis (EDA), predictive modeling, and business recommendations.

Repository Structure

Data-Masters-Challenge-Team-3/

├── scripts/

│   ├── data_cleaning.py                # Cleans raw energy consumption data

│   ├── energy_analysis.py              # Analyzes energy usage patterns

│   ├── summarize_data_issues.py        # Summarizes issues in raw data

│   ├── summarize_cleaned_data_issues.py # Summarizes issues in cleaned data

│   ├── Data_Masters_Challenge_Olympiad_Team_3_Predictive_Models.ipynb # Predictive modeling notebook

├── reports/

│   ├── Data_Cleaning_Report.pdf        # Data cleaning process and results

│   ├── Exploratory Data Analyis.pdf     # Select Exploratory data analysis findings w/ visualizations

│   ├── Business Insights Report.pdf      # Business recommendations for energy efficiency

│   ├── EDA Visualizations.pbix        # EDA interactive dashboards - comprehensive

├── README.md                           # This file

Data Notes





No Full Data Included: Full datasets (e.g., electricity.csv, cleaned_electricity-0.csv) are not included due to size constraints. Results and methodologies are documented in reports/.



Data Context: Scripts operate on energy consumption data with timestamps and building-specific usage (e.g., Eagle_education_Brooke). 

Installation





Clone the repository:

git clone https://github.com/yourusername/Data-Masters-Challenge-Team-3.git



Install dependencies (on Windows, use Command Prompt or PowerShell):

pip install pandas matplotlib xgboost scikit-learn jupyter

Usage





Run Python scripts from the scripts/ directory:

python scripts/data_cleaning.py
python scripts/energy_analysis.py
python scripts/summarize_data_issues.py
python scripts/summarize_cleaned_data_issues.py



Open the predictive modeling notebook in Jupyter Notebook:

jupyter notebook scripts/Data_Masters_Challenge_Olympiad_Team_3_Predictive_Models.ipynb



Review reports in reports/ for detailed findings and recommendations:





Data Cleaning Report.pdf: Details on data quality improvements.



Exploratory Data Analysis.pdf: Visualizations and trends in energy consumption.



Business Insights Report.pdf: Strategies for 20% energy waste reduction.

Submission Components





Data Cleaning Report (reports/Data_Cleaning_Report.pdf): Describes handling of missing values (e.g., reduced from 4.74% to 0.13%) and data transformation.



Exploratory Data Analysis (reports/Exploratory Data Analysis.pdf): Analyzes consumption patterns to identify waste (e.g., HVAC overuse).

EDA Visualizations (reports/EDA Visualizations.pbix): Comprehensive dashboard, complete with visualizations of key findings concerning energy usage.



Predictive Modeling (scripts/Data_Masters_Challenge_Olympiad_Team_3_Predictive_Models.ipynb): Implements models (e.g., XGBoost) to predict consumption and target inefficiencies (e.g., 225,000 kWh/building savings).



Business Insight Report (reports/Business Insights Report.pdf): Recommends building upgrades and operational changes (e.g., potential 237,122,120 kWh savings).



Code: All scripts and notebooks in scripts/.
