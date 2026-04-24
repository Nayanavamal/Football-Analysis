import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, shapiro, chi2_contingency
#hello
# -------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_excel("football_updated.xlsx")

# -------------------------------
# BASIC INFO
# -------------------------------
print("First 5 Rows:\n",df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["GA"] = df["Goals"] + df["Assists"]

# -------------------------------
# GOALS DISTRIBUTION (AREA + LINE)
# -------------------------------
goal_counts = df["Goals"].value_counts().sort_index()

plt.figure()
plt.fill_between(goal_counts.index, goal_counts.values, alpha=0.3)
plt.plot(goal_counts.index, goal_counts.values)

plt.title("Goals Distribution (Area + Line)")
plt.xlabel("Goals")
plt.ylabel("Number of Players")
plt.grid(True)
plt.show()

# -------------------------------
# HISTOGRAM (SALARY)
# -------------------------------
plt.figure()
plt.hist(df["Salary_M_Euro"], bins=10)

plt.title("Salary Distribution")
plt.xlabel("Salary (Million Euro)")
plt.ylabel("Number of Players")
plt.grid(True)
plt.show()

# -------------------------------
# BOXPLOT (OUTLIERS)
# -------------------------------
plt.figure()
sns.boxplot(y=df["Salary_M_Euro"])
plt.title("Salary Outliers")
plt.ylabel("Salary")
plt.show()

plt.figure()
sns.boxplot(y=df["Goals"])
plt.title("Goals Outliers")
plt.ylabel("Goals")
plt.show()

# -------------------------------
# OUTLIER DETECTION (IQR) SALARY
# -------------------------------
Q1 = df["Salary_M_Euro"].quantile(0.25)
Q3 = df["Salary_M_Euro"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Salary Outliers below:", lower)
print("Salary Outliers above:", upper)

outliers = df[(df["Salary_M_Euro"] < lower) | (df["Salary_M_Euro"] > upper)]

print("\nSalary Outliers:\n")
print(outliers[["Player_ID", "Salary_M_Euro"]])

# -------------------------------
# OUTLIER DETECTION (IQR) GOALS
# -------------------------------
Q1 = df["Goals"].quantile(0.25)
Q3 = df["Goals"].quantile(0.75)
IQR = Q3 - Q1

lower1 = Q1 - 1.5 * IQR
upper1 = Q3 + 1.5 * IQR

print("Goal Outliers below:", lower1)
print("Goal Outliers above:", upper1)
print("Note: Forwards naturally score more, so high values are expected.")

# -------------------------------
# SCATTER (xG vs Goals)
# -------------------------------
plt.figure()
plt.scatter(df["xG"], df["Goals"])

plt.title("xG vs Goals")
plt.xlabel("Expected Goals (xG)")
plt.ylabel("Actual Goals")
plt.grid(True)
plt.show()

# -------------------------------
# POSITION-WISE ANALYSIS
# -------------------------------
position_avg = df.groupby("Position")[["Goals", "Assists"]].mean()

print("\nPosition-wise Average:\n", position_avg)

position_avg.plot(kind="bar")
plt.title("Average Goals & Assists by Position")
plt.xlabel("Position")
plt.ylabel("Average Value")
plt.grid(True)
plt.show()

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
numeric_df = df.select_dtypes(include=[np.number]).copy()
numeric_df.drop(columns=["Player_ID"], inplace=True, errors="ignore")

corr = numeric_df.corr()

plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f")

plt.title("Correlation Heatmap (Excluding Player_ID)")
plt.show()

# -------------------------------
# T-TEST (FW vs MF)
# -------------------------------
fw = df[df["Position"] == "FW"]["Goals"]
mf = df[df["Position"] == "MF"]["Goals"]

t_stat, p_value = ttest_ind(fw, mf)

print("\nT-Test Results (FW vs MF Goals)")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant difference in goals")
else:
    print("No significant difference")

# -------------------------------
# SHAPIRO TEST (NORMALITY)
# -------------------------------
stat, p = shapiro(df["Goals"])

print("\nShapiro Test (Goals)")
print("P-value:", p)

if p > 0.05:
    print("Data is normally distributed")
else:
    print("Data is NOT normally distributed")

# -------------------------------
# CHI-SQUARE TEST
# -------------------------------
table = pd.crosstab(df["Team"], df["Position"])

chi2, p, dof, expected = chi2_contingency(table)

print("\nChi-Square Test")
print("P-value:", p)

# -------------------------------
# TOP 5 PLAYERS
# -------------------------------
top_players = df.sort_values(by="GA", ascending=False).head(5)

print("\nTop 5 Players (Goals + Assists):\n")
print(top_players[["Player_ID", "Team", "Goals", "Assists", "GA"]])

# ============================================================
#  ADDITIONAL VISUALIZATIONS (SYLLABUS ONLY)
# ============================================================

# -------------------------------
# AVERAGE SALARY BY POSITION (PIE CHART)
# -------------------------------

avg_salary = df.groupby("Position")["Salary_M_Euro"].mean()

plt.figure()
plt.pie(
    avg_salary,
    labels=avg_salary.index,
    autopct='%1.1f%%'
)

plt.title("Average Salary Distribution by Position")

plt.show()

# -------------------------------
# TEAM-WISE TOTAL GOALS
# -------------------------------
team_goals = df.groupby("Team")["Goals"].sum().sort_values(ascending=False)

plt.figure()
team_goals.plot(kind="bar")

plt.title("Total Goals by Team")
plt.xlabel("Team")
plt.ylabel("Total Goals")
plt.grid(True)
plt.show()

# -------------------------------
# TEAM-WISE AVG DISTANCE
# -------------------------------
if "Distance_Covered" in df.columns:
    team_distance = df.groupby("Team")["Distance_Covered"].mean().sort_values(ascending=False)

    plt.figure()
    team_distance.plot(kind="bar")

    plt.title("Average Distance Covered by Team")
    plt.xlabel("Team")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.show()

# -------------------------------
# TEAM vs POSITION (STACKED BAR)
# -------------------------------
team_position = df.groupby(["Team", "Position"])["Goals"].sum().unstack()

team_position.plot(kind="bar", stacked=True)

plt.title("Team-wise Goals by Position")
plt.xlabel("Team")
plt.ylabel("Goals")
plt.show()

# -------------------------------
# SALARY vs GOALS
# -------------------------------
plt.figure()
plt.scatter(df["Salary_M_Euro"], df["Goals"])

plt.title("Salary vs Goals")
plt.xlabel("Salary (Million Euro)")
plt.ylabel("Goals")
plt.grid(True)
plt.show()

# -------------------------------
# AGE vs PERFORMANCE BY POSITION
# -------------------------------
plt.figure()
sns.scatterplot(x="Age", y="GA", hue="Position", data=df)

plt.title("Age vs Performance by Position")
plt.xlabel("Age")
plt.ylabel("GA")
plt.grid(True)
plt.show()

# -------------------------------
# AGE GROUP ANALYSIS
# -------------------------------
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[18, 22, 26, 30, 35],
    labels=["18-22", "23-26", "27-30", "31-35"]
)

age_group_perf = df.groupby("Age_Group")["GA"].mean()

plt.figure()
age_group_perf.plot(kind="bar")

plt.title("Average Performance by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average GA")
plt.grid(True)
plt.show()

# -------------------------------
# POSITION vs PERFORMANCE (BOXPLOT)
# -------------------------------
plt.figure()
sns.boxplot(x="Position", y="GA", data=df)

plt.title("Performance Distribution by Position")
plt.xlabel("Position")
plt.ylabel("GA")
plt.show()

# -------------------------------
# TEAM vs POSITION HEATMAP (GOALS)
# -------------------------------

team_position_goals = df.pivot_table(
    values="Goals",
    index="Team",
    columns="Position",
    aggfunc="sum"
)

plt.figure()
sns.heatmap(team_position_goals, annot=True, fmt=".0f")

plt.title("Team vs Position Goals Heatmap")
plt.xlabel("Position")
plt.ylabel("Team")

plt.show()
