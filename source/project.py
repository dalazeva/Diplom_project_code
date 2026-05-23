#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("risk_decoded.csv")

df.head()


# In[2]:


print("Размер набора данных:", df.shape)

display(df.head())
display(df.info())
display(df.describe(include="all"))


# In[3]:


missing = df.isna().sum().sort_values(ascending=False)

missing_table = pd.DataFrame({
    "Пропущенные значения": missing,
    "Доля пропусков, %": (missing / len(df) * 100).round(2)
})

display(missing_table)


# In[4]:


unique_values = pd.DataFrame({
    "Количество уникальных значений": df.nunique(),
    "Тип данных": df.dtypes
})

display(unique_values)


# In[5]:


target_col = "cancer"
weight_col = "count"
split_col = "training"

all_columns = df.columns.tolist()

categorical_cols = [
    col for col in df.columns
    if col not in [target_col, weight_col]
]

feature_cols = [
    col for col in categorical_cols
    if col not in [split_col, "invasive"]
]

print("Целевая переменная:", target_col)
print("Весовой столбец:", weight_col)
print("Столбец разбиения:", split_col)
print("Признаки для анализа:")
print(feature_cols)


# In[6]:


target_counts = df[target_col].value_counts()

display(target_counts)


# In[7]:


plt.figure(figsize=(6, 4))
target_counts.plot(kind="bar")
plt.title("Распределение целевой переменной без учета весов")
plt.xlabel("Наличие рака")
plt.ylabel("Количество строк")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[8]:


target_weighted = df.groupby(target_col)[weight_col].sum().sort_values(ascending=False)

target_weighted_table = pd.DataFrame({
    "Количество наблюдений": target_weighted,
    "Доля, %": (target_weighted / target_weighted.sum() * 100).round(2)
})

display(target_weighted_table)


# In[9]:


plt.figure(figsize=(6, 4))
target_weighted.plot(kind="bar")
plt.title("Распределение целевой переменной с учетом count")
plt.xlabel("Наличие рака")
plt.ylabel("Количество наблюдений")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[10]:


split_weighted = df.groupby(split_col)[weight_col].sum()

split_table = pd.DataFrame({
    "Количество наблюдений": split_weighted,
    "Доля, %": (split_weighted / split_weighted.sum() * 100).round(2)
})

display(split_table)


# In[11]:


plt.figure(figsize=(6, 4))
split_weighted.plot(kind="bar")
plt.title("Распределение наблюдений по обучающей и валидационной выборкам")
plt.xlabel("Выборка")
plt.ylabel("Количество наблюдений")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[12]:


split_target = pd.crosstab(
    df[split_col],
    df[target_col],
    values=df[weight_col],
    aggfunc="sum"
).fillna(0)

split_target_percent = split_target.div(split_target.sum(axis=1), axis=0) * 100

display(split_target)
display(split_target_percent.round(2))


# In[13]:


split_target_percent.plot(kind="bar", stacked=True, figsize=(7, 4))
plt.title("Доля классов cancer в обучающей и валидационной выборках")
plt.xlabel("Выборка")
plt.ylabel("Доля, %")
plt.xticks(rotation=0)
plt.legend(title="cancer")
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[14]:


def plot_weighted_distribution(data, column, weight_col="count"):
    values = data.groupby(column)[weight_col].sum().sort_values(ascending=False)

    plt.figure(figsize=(9, 4))
    values.plot(kind="bar")
    plt.title(f"Распределение признака: {column}")
    plt.xlabel(column)
    plt.ylabel("Количество наблюдений")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    table = pd.DataFrame({
        "Количество наблюдений": values,
        "Доля, %": (values / values.sum() * 100).round(2)
    })

    display(table)


# In[15]:


for col in feature_cols:
    plot_weighted_distribution(df, col, weight_col)


# In[16]:


def cancer_rate_by_feature(data, column, target_col="cancer", weight_col="count", positive_class="yes"):
    table = pd.crosstab(
        data[column],
        data[target_col],
        values=data[weight_col],
        aggfunc="sum"
    ).fillna(0)

    if positive_class not in table.columns:
        table[positive_class] = 0

    table["total"] = table.sum(axis=1)
    table["cancer_rate"] = table[positive_class] / table["total"] * 100

    table = table.sort_values("cancer_rate", ascending=False)

    display(table.round(3))

    plt.figure(figsize=(9, 4))
    table["cancer_rate"].plot(kind="bar")
    plt.title(f"Доля случаев cancer = yes по признаку: {column}")
    plt.xlabel(column)
    plt.ylabel("Доля cancer = yes, %")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return table


# In[17]:


cancer_rate_tables = {}

for col in feature_cols:
    cancer_rate_tables[col] = cancer_rate_by_feature(
        df,
        column=col,
        target_col=target_col,
        weight_col=weight_col,
        positive_class="yes"
    )


# In[18]:


def plot_feature_target_distribution(data, column, target_col="cancer", weight_col="count"):
    table = pd.crosstab(
        data[column],
        data[target_col],
        values=data[weight_col],
        aggfunc="sum"
    ).fillna(0)

    table_percent = table.div(table.sum(axis=1), axis=0) * 100

    table_percent.plot(kind="bar", stacked=True, figsize=(9, 4))
    plt.title(f"Структура классов cancer по признаку: {column}")
    plt.xlabel(column)
    plt.ylabel("Доля, %")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=target_col)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    display(table)
    display(table_percent.round(2))


# In[19]:


for col in feature_cols:
    plot_feature_target_distribution(
        df,
        column=col,
        target_col=target_col,
        weight_col=weight_col
    )


# In[20]:


invasive_table = pd.crosstab(
    df["invasive"],
    df[target_col],
    values=df[weight_col],
    aggfunc="sum"
).fillna(0)

invasive_percent = invasive_table.div(invasive_table.sum(axis=1), axis=0) * 100

display(invasive_table)
display(invasive_percent.round(2))


# In[21]:


invasive_percent.plot(kind="bar", stacked=True, figsize=(7, 4))
plt.title("Связь признака invasive с целевой переменной cancer")
plt.xlabel("invasive")
plt.ylabel("Доля, %")
plt.xticks(rotation=0)
plt.legend(title="cancer")
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[22]:


summary_rows = []

for col in feature_cols:
    table = pd.crosstab(
        df[col],
        df[target_col],
        values=df[weight_col],
        aggfunc="sum"
    ).fillna(0)

    if "yes" not in table.columns:
        table["yes"] = 0

    table["total"] = table.sum(axis=1)
    table["cancer_rate_percent"] = table["yes"] / table["total"] * 100

    for category, row in table.iterrows():
        summary_rows.append({
            "feature": col,
            "category": category,
            "total": row["total"],
            "cancer_yes": row["yes"],
            "cancer_rate_percent": row["cancer_rate_percent"]
        })

cancer_rate_summary = pd.DataFrame(summary_rows)

cancer_rate_summary = cancer_rate_summary.sort_values(
    by="cancer_rate_percent",
    ascending=False
)

display(cancer_rate_summary.head(30))


# In[23]:


top_categories = cancer_rate_summary[
    cancer_rate_summary["total"] >= 100
].head(20)

plt.figure(figsize=(10, 6))
plt.barh(
    top_categories["feature"] + ": " + top_categories["category"],
    top_categories["cancer_rate_percent"]
)
plt.title("Категории с наибольшей долей cancer = yes")
plt.xlabel("Доля cancer = yes, %")
plt.ylabel("Категория")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# In[24]:


from scipy.stats import chi2_contingency

def cramers_v_weighted(data, feature, target, weight_col):
    table = pd.crosstab(
        data[feature],
        data[target],
        values=data[weight_col],
        aggfunc="sum"
    ).fillna(0)

    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()

    r, k = table.shape

    if n == 0:
        return np.nan

    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))


# In[25]:


cramers_results = []

for col in feature_cols:
    value = cramers_v_weighted(df, col, target_col, weight_col)
    cramers_results.append({
        "feature": col,
        "cramers_v": value
    })

cramers_df = pd.DataFrame(cramers_results).sort_values(
    by="cramers_v",
    ascending=False
)

display(cramers_df)


# In[26]:


plt.figure(figsize=(8, 5))
plt.barh(cramers_df["feature"], cramers_df["cramers_v"])
plt.title("Связь признаков с целевой переменной cancer по Cramer's V")
plt.xlabel("Cramer's V")
plt.ylabel("Признак")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# In[27]:


print("Количество строк в исходном наборе:", len(df))
print("Суммарное количество наблюдений с учетом count:", df[weight_col].sum())
print("Количество признаков для будущего обучения:", len(feature_cols))
print("Признаки:")
print(feature_cols)


# In[28]:


final_check = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum().values,
    "unique_values": df.nunique().values
})

display(final_check)


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv("risk_decoded.csv")

target_col = "cancer"
positive_class = "yes"
weight_col = "count"

feature_cols = [
    "menopaus",
    "agegrp",
    "density",
    "race",
    "Hispanic",
    "bmi",
    "agefirst",
    "nrelbc",
    "brstproc",
    "lastmamm",
    "surgmeno",
    "hrt"
]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# In[30]:


def weighted_cancer_rate(data, group_cols):
    table = data.groupby(group_cols + [target_col])[weight_col].sum().reset_index()

    pivot = table.pivot_table(
        index=group_cols,
        columns=target_col,
        values=weight_col,
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    if positive_class not in pivot.columns:
        pivot[positive_class] = 0

    class_cols = [col for col in pivot.columns if col not in group_cols]

    pivot["total"] = pivot[class_cols].sum(axis=1)
    pivot["cancer_yes"] = pivot[positive_class]
    pivot["cancer_rate"] = pivot["cancer_yes"] / pivot["total"] * 100

    return pivot


# In[31]:


age_density = weighted_cancer_rate(df, ["agegrp", "density"])

heatmap_data = age_density.pivot(
    index="agegrp",
    columns="density",
    values="cancer_rate"
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="Reds",
    linewidths=0.5
)

plt.title("Доля случаев cancer = yes по возрасту и плотности ткани, %")
plt.xlabel("Плотность ткани")
plt.ylabel("Возрастная группа")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()


# In[32]:


age_bmi = weighted_cancer_rate(df, ["agegrp", "bmi"])

heatmap_data = age_bmi.pivot(
    index="agegrp",
    columns="bmi",
    values="cancer_rate"
)

plt.figure(figsize=(11, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="Oranges",
    linewidths=0.5
)

plt.title("Доля случаев cancer = yes по возрасту и BMI, %")
plt.xlabel("BMI")
plt.ylabel("Возрастная группа")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# In[33]:


bubble_data = weighted_cancer_rate(df, ["agegrp", "density"])

plt.figure(figsize=(13, 6))

scatter = sns.scatterplot(
    data=bubble_data,
    x="agegrp",
    y="density",
    size="total",
    hue="cancer_rate",
    sizes=(50, 1200),
    palette="Reds",
    alpha=0.75
)

plt.title("Возраст, плотность ткани и доля cancer = yes")
plt.xlabel("Возрастная группа")
plt.ylabel("Плотность ткани")
plt.xticks(rotation=45)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title="Показатели"
)
plt.tight_layout()
plt.show()


# In[34]:


overall_rate = (
    df.loc[df[target_col] == positive_class, weight_col].sum()
    / df[weight_col].sum()
)

risk_rows = []

for feature in feature_cols:
    temp = weighted_cancer_rate(df, [feature])

    for _, row in temp.iterrows():
        risk_rows.append({
            "feature": feature,
            "category": row[feature],
            "total": row["total"],
            "cancer_rate": row["cancer_rate"],
            "relative_risk": (row["cancer_rate"] / 100) / overall_rate
        })

risk_df = pd.DataFrame(risk_rows)

risk_df = risk_df[risk_df["total"] >= 500]
risk_df = risk_df.sort_values("relative_risk", ascending=False)

display(risk_df.head(30))


# In[35]:


top_risk = risk_df.head(25).copy()
top_risk["label"] = top_risk["feature"] + ": " + top_risk["category"]

plt.figure(figsize=(10, 8))
sns.barplot(
    data=top_risk,
    x="relative_risk",
    y="label",
    orient="h"
)

plt.axvline(1, color="black", linestyle="--")
plt.title("Категории с наибольшим относительным риском cancer = yes")
plt.xlabel("Относительный риск")
plt.ylabel("Категория")
plt.tight_layout()
plt.show()


# In[36]:


profile_cols = ["agegrp", "density", "nrelbc", "brstproc"]

profile_risk = weighted_cancer_rate(df, profile_cols)

profile_risk = profile_risk[profile_risk["total"] >= 1000]
profile_risk = profile_risk.sort_values("cancer_rate", ascending=False)

profile_risk["profile"] = (
    "age: " + profile_risk["agegrp"].astype(str)
    + " | density: " + profile_risk["density"].astype(str)
    + " | relatives: " + profile_risk["nrelbc"].astype(str)
    + " | procedure: " + profile_risk["brstproc"].astype(str)
)

display(profile_risk.head(20))


# In[37]:


top_profiles = profile_risk.head(15)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_profiles,
    x="cancer_rate",
    y="profile",
    orient="h"
)

plt.title("Профили пациенток с наибольшей долей cancer = yes")
plt.xlabel("Доля cancer = yes, %")
plt.ylabel("Профиль")
plt.tight_layout()
plt.show()


# In[38]:


def plot_100_stacked_bar(data, feature):
    table = pd.crosstab(
        data[feature],
        data[target_col],
        values=data[weight_col],
        aggfunc="sum"
    ).fillna(0)

    table_percent = table.div(table.sum(axis=1), axis=0) * 100

    table_percent.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5),
        colormap="Set2"
    )

    plt.title(f"Структура классов cancer по признаку {feature}, %")
    plt.xlabel(feature)
    plt.ylabel("Доля, %")
    plt.xticks(rotation=40, ha="right")
    plt.legend(title="cancer")
    plt.tight_layout()
    plt.show()


# In[39]:


important_features = ["agegrp", "density", "bmi", "nrelbc", "brstproc", "lastmamm"]

for feature in important_features:
    plot_100_stacked_bar(df, feature)


# In[40]:


def cramers_v(data, col1, col2, weight_col):
    table = pd.crosstab(
        data[col1],
        data[col2],
        values=data[weight_col],
        aggfunc="sum"
    ).fillna(0)

    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()

    r, k = table.shape

    if n == 0 or min(r - 1, k - 1) == 0:
        return np.nan

    return np.sqrt(chi2 / (n * min(r - 1, k - 1)))


# In[41]:


cramers_matrix = pd.DataFrame(
    index=feature_cols,
    columns=feature_cols,
    dtype=float
)

for col1 in feature_cols:
    for col2 in feature_cols:
        cramers_matrix.loc[col1, col2] = cramers_v(df, col1, col2, weight_col)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cramers_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    linewidths=0.5,
    vmin=0,
    vmax=1
)

plt.title("Матрица связи категориальных признаков по Cramer's V")
plt.xlabel("Признаки")
plt.ylabel("Признаки")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[42]:


target_association = []

for feature in feature_cols:
    value = cramers_v(df, feature, target_col, weight_col)

    target_association.append({
        "feature": feature,
        "cramers_v_with_cancer": value
    })

target_association = pd.DataFrame(target_association)
target_association = target_association.sort_values(
    "cramers_v_with_cancer",
    ascending=False
)

display(target_association)


# In[43]:


plt.figure(figsize=(9, 6))
sns.barplot(
    data=target_association,
    x="cramers_v_with_cancer",
    y="feature",
    orient="h"
)

plt.title("Сила связи признаков с целевой переменной cancer")
plt.xlabel("Cramer's V")
plt.ylabel("Признак")
plt.tight_layout()
plt.show()


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("risk_decoded.csv")

positive_df = df[df["cancer"] == "yes"].copy()

heatmap_counts = positive_df.pivot_table(
    index="agegrp",
    columns="density",
    values="count",
    aggfunc="sum",
    fill_value=0
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_counts,
    annot=True,
    fmt=".0f",
    cmap="OrRd",
    linewidths=0.5
)

plt.title("Тепловая карта количества положительных случаев cancer = yes")
plt.xlabel("Плотность ткани")
plt.ylabel("Возрастная группа")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[45]:


import pandas as pd
import numpy as np

df = pd.read_csv("risk_decoded.csv")

display(df.head())
print(df.shape)


# In[46]:


print("Размер набора данных:", df.shape)

print("\nТипы данных:")
print(df.dtypes)

print("\nКоличество пропусков:")
print(df.isna().sum())

print("\nЗначения целевой переменной:")
print(df["cancer"].value_counts())

print("\nРаспределение с учетом count:")
print(df.groupby("cancer")["count"].sum())


# In[47]:


target_col = "cancer"
weight_col = "count"
split_col = "training"

leakage_cols = ["invasive"]

service_cols = [
    target_col,
    weight_col,
    split_col
] + leakage_cols

feature_cols = [
    col for col in df.columns
    if col not in service_cols
]

print("Целевая переменная:", target_col)
print("Весовой столбец:", weight_col)
print("Столбец разбиения:", split_col)
print("Исключенные признаки:", leakage_cols)
print("Признаки для моделирования:")
print(feature_cols)


# In[48]:


train_df = df[df[split_col] == "training"].copy()
valid_df = df[df[split_col] == "validation"].copy()

print("Размер обучающей выборки:", train_df.shape)
print("Размер валидационной выборки:", valid_df.shape)


# In[49]:


X_train = train_df[feature_cols].copy()
X_valid = valid_df[feature_cols].copy()

y_train = train_df[target_col].map({
    "no": 0,
    "yes": 1
})

y_valid = valid_df[target_col].map({
    "no": 0,
    "yes": 1
})

sample_weight_train = train_df[weight_col].copy()
sample_weight_valid = valid_df[weight_col].copy()

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)

print("\nРаспределение классов в обучающей выборке:")
print(y_train.value_counts())

print("\nРаспределение классов в валидационной выборке:")
print(y_valid.value_counts())


# In[50]:


categorical_features = X_train.columns.tolist()

for col in categorical_features:
    print(col)
    print("Количество категорий:", X_train[col].nunique())
    print("Категории:", sorted(X_train[col].unique()))
    print()


# In[51]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder, categorical_features)
    ],
    remainder="drop"
)


# In[52]:


X_train_encoded = preprocessor.fit_transform(X_train)
X_valid_encoded = preprocessor.transform(X_valid)

print("Размер X_train после кодирования:", X_train_encoded.shape)
print("Размер X_valid после кодирования:", X_valid_encoded.shape)


# In[53]:


encoded_feature_names = preprocessor.get_feature_names_out()

encoded_feature_names = [
    name.replace("cat__", "")
    for name in encoded_feature_names
]

print("Количество признаков после кодирования:", len(encoded_feature_names))
print(encoded_feature_names[:20])


# In[54]:


X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=encoded_feature_names,
    index=X_train.index
)

X_valid_encoded_df = pd.DataFrame(
    X_valid_encoded,
    columns=encoded_feature_names,
    index=X_valid.index
)

display(X_train_encoded_df.head())
display(X_valid_encoded_df.head())


# In[55]:


print("Исходное количество признаков:", len(feature_cols))
print("Количество признаков после One-Hot Encoding:", X_train_encoded_df.shape[1])

print("\nРазмеры итоговых объектов:")
print("X_train_encoded_df:", X_train_encoded_df.shape)
print("X_valid_encoded_df:", X_valid_encoded_df.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)
print("sample_weight_train:", sample_weight_train.shape)
print("sample_weight_valid:", sample_weight_valid.shape)


# In[56]:


train_class_balance = train_df.groupby(target_col)[weight_col].sum()
valid_class_balance = valid_df.groupby(target_col)[weight_col].sum()

train_class_balance_percent = train_class_balance / train_class_balance.sum() * 100
valid_class_balance_percent = valid_class_balance / valid_class_balance.sum() * 100

print("Баланс классов в обучающей выборке с учетом count:")
display(pd.DataFrame({
    "count": train_class_balance,
    "percent": train_class_balance_percent.round(2)
}))

print("Баланс классов в валидационной выборке с учетом count:")
display(pd.DataFrame({
    "count": valid_class_balance,
    "percent": valid_class_balance_percent.round(2)
}))


# In[57]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_data(path):
    df = pd.read_csv(path)

    target_col = "cancer"
    weight_col = "count"
    split_col = "training"

    leakage_cols = ["invasive"]

    service_cols = [
        target_col,
        weight_col,
        split_col
    ] + leakage_cols

    feature_cols = [
        col for col in df.columns
        if col not in service_cols
    ]

    train_df = df[df[split_col] == "training"].copy()
    valid_df = df[df[split_col] == "validation"].copy()

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()

    y_train = train_df[target_col].map({
        "no": 0,
        "yes": 1
    })

    y_valid = valid_df[target_col].map({
        "no": 0,
        "yes": 1
    })

    sample_weight_train = train_df[weight_col].copy()
    sample_weight_valid = valid_df[weight_col].copy()

    categorical_features = X_train.columns.tolist()

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical_features)
        ],
        remainder="drop"
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_valid_encoded = preprocessor.transform(X_valid)

    encoded_feature_names = preprocessor.get_feature_names_out()

    encoded_feature_names = [
        name.replace("cat__", "")
        for name in encoded_feature_names
    ]

    X_train_encoded = pd.DataFrame(
        X_train_encoded,
        columns=encoded_feature_names,
        index=X_train.index
    )

    X_valid_encoded = pd.DataFrame(
        X_valid_encoded,
        columns=encoded_feature_names,
        index=X_valid.index
    )

    return {
        "df": df,
        "train_df": train_df,
        "valid_df": valid_df,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_train_encoded": X_train_encoded,
        "X_valid_encoded": X_valid_encoded,
        "y_train": y_train,
        "y_valid": y_valid,
        "sample_weight_train": sample_weight_train,
        "sample_weight_valid": sample_weight_valid,
        "feature_cols": feature_cols,
        "encoded_feature_names": encoded_feature_names,
        "preprocessor": preprocessor
    }


# In[58]:


data = prepare_data("risk_decoded.csv")

df = data["df"]

X_train = data["X_train"]
X_valid = data["X_valid"]

X_train_encoded = data["X_train_encoded"]
X_valid_encoded = data["X_valid_encoded"]

y_train = data["y_train"]
y_valid = data["y_valid"]

sample_weight_train = data["sample_weight_train"]
sample_weight_valid = data["sample_weight_valid"]

feature_cols = data["feature_cols"]
encoded_feature_names = data["encoded_feature_names"]
preprocessor = data["preprocessor"]

print("Готово.")

print("X_train_encoded:", X_train_encoded.shape)
print("X_valid_encoded:", X_valid_encoded.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)


# In[59]:


display(X_train_encoded.head())

print("Есть ли пропуски в X_train_encoded:")
print(X_train_encoded.isna().sum().sum())

print("Есть ли пропуски в X_valid_encoded:")
print(X_valid_encoded.isna().sum().sum())

print("Уникальные значения y_train:")
print(y_train.unique())

print("Уникальные значения y_valid:")
print(y_valid.unique())


# In[60]:


X_train_encoded.to_csv("X_train_encoded.csv", index=False)
X_valid_encoded.to_csv("X_valid_encoded.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_valid.to_csv("y_valid.csv", index=False)

sample_weight_train.to_csv("sample_weight_train.csv", index=False)
sample_weight_valid.to_csv("sample_weight_valid.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


# In[62]:


DATA_PATH = "risk.txt"

columns = [
    "menopaus",
    "agegrp",
    "density",
    "race",
    "Hispanic",
    "bmi",
    "agefirst",
    "nrelbc",
    "brstproc",
    "lastmamm",
    "surgmeno",
    "hrt",
    "invasive",
    "cancer",
    "training",
    "count"
]

risk = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)

if risk.shape[1] != 16:
    raise ValueError(f"Ожидалось 16 колонок, но найдено {risk.shape[1]}")

risk.columns = columns

for col in columns:
    risk[col] = risk[col].astype(int)

risk.head()


# In[63]:


print("Размер агрегированного датасета:", risk.shape)
print("Количество маммограмм с учётом count:", risk["count"].sum())
print("Количество уникальных комбинаций признаков:", len(risk))

weighted_cancer_rate = np.average(risk["cancer"], weights=risk["count"])

print(f"Взвешенная доля случаев cancer = 1: {weighted_cancer_rate:.5f}")
print(f"В процентах: {weighted_cancer_rate * 100:.3f}%")


# In[64]:


risk["training"].value_counts()


# In[65]:


train_df = risk[risk["training"] == 1].copy()
valid_df = risk[risk["training"] == 0].copy()

print("Train:", train_df.shape)
print("Validation:", valid_df.shape)

print("Train, weighted N:", train_df["count"].sum())
print("Validation, weighted N:", valid_df["count"].sum())


# In[66]:


target = "cancer"

drop_columns = [
    "invasive",   # не используем как признак, так как это другой вариант целевой переменной
    "cancer",     # целевая переменная
    "training",   # технический признак разбиения
    "count"       # вес наблюдения
]

features = [col for col in columns if col not in drop_columns]

X_train = train_df[features]
y_train = train_df[target]
w_train = train_df["count"]

X_valid = valid_df[features]
y_valid = valid_df[target]
w_valid = valid_df["count"]

features


# In[67]:


class_weight_0 = w_train.sum() / (2 * w_train[y_train == 0].sum())
class_weight_1 = w_train.sum() / (2 * w_train[y_train == 1].sum())

print("Вес класса 0:", class_weight_0)
print("Вес класса 1:", class_weight_1)

w_train_balanced = w_train.copy().astype(float)

w_train_balanced[y_train == 0] = w_train[y_train == 0] * class_weight_0
w_train_balanced[y_train == 1] = w_train[y_train == 1] * class_weight_1

print("Суммарный исходный вес класса 0:", w_train[y_train == 0].sum())
print("Суммарный исходный вес класса 1:", w_train[y_train == 1].sum())

print("Суммарный сбалансированный вес класса 0:", w_train_balanced[y_train == 0].sum())
print("Суммарный сбалансированный вес класса 1:", w_train_balanced[y_train == 1].sum())


# In[68]:


balance_before_after = pd.DataFrame({
    "class": ["cancer=0", "cancer=1"],
    "before_balancing": [
        w_train[y_train == 0].sum(),
        w_train[y_train == 1].sum()
    ],
    "after_balancing": [
        w_train_balanced[y_train == 0].sum(),
        w_train_balanced[y_train == 1].sum()
    ]
})

balance_before_after


# In[ ]:





# In[ ]:





# In[69]:


try:
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse=True
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder, features)
    ],
    remainder="drop"
)

def make_logistic_model(C=1.0):
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=42
            ))
        ]
    )
    
    return model


# In[70]:


baseline_model = make_logistic_model(C=1.0)

baseline_model.fit(
    X_train,
    y_train,
    model__sample_weight=w_train_balanced
)

baseline_valid_prob = baseline_model.predict_proba(X_valid)[:, 1]

baseline_auc = roc_auc_score(
    y_valid,
    baseline_valid_prob,
    sample_weight=w_valid
)

baseline_logloss = log_loss(
    y_valid,
    baseline_valid_prob,
    sample_weight=w_valid,
    labels=[0, 1]
)

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
print(f"Baseline Log Loss: {baseline_logloss:.6f}")


# In[71]:


def make_balanced_weights(y, weights):
    class_0_sum = weights[y == 0].sum()
    class_1_sum = weights[y == 1].sum()
    total_sum = weights.sum()
    
    class_weight_0 = total_sum / (2 * class_0_sum)
    class_weight_1 = total_sum / (2 * class_1_sum)
    
    balanced_weights = weights.copy().astype(float)
    balanced_weights[y == 0] = weights[y == 0] * class_weight_0
    balanced_weights[y == 1] = weights[y == 1] * class_weight_1
    
    return balanced_weights


C_grid = np.logspace(-3, 3, 7)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for C in C_grid:
    fold_auc = []
    fold_logloss = []
    fold_brier = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        w_tr = w_train.iloc[train_idx]
        
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        w_val = w_train.iloc[val_idx]
        
        w_tr_balanced = make_balanced_weights(y_tr, w_tr)
        
        model = make_logistic_model(C=C)
        
        model.fit(
            X_tr,
            y_tr,
            model__sample_weight=w_tr_balanced
        )
        
        val_prob = model.predict_proba(X_val)[:, 1]
        
        fold_auc.append(
            roc_auc_score(
                y_val,
                val_prob,
                sample_weight=w_val
            )
        )
        
        fold_logloss.append(
            log_loss(
                y_val,
                val_prob,
                sample_weight=w_val,
                labels=[0, 1]
            )
        )
        
        fold_brier.append(
            brier_score_loss(
                y_val,
                val_prob,
                sample_weight=w_val
            )
        )
    
    cv_results.append({
        "C": C,
        "mean_auc": np.mean(fold_auc),
        "std_auc": np.std(fold_auc),
        "mean_log_loss": np.mean(fold_logloss),
        "std_log_loss": np.std(fold_logloss),
        "mean_brier": np.mean(fold_brier),
        "std_brier": np.std(fold_brier)
    })

cv_results = pd.DataFrame(cv_results)

cv_results


# In[72]:


cv_results_sorted = cv_results.sort_values(
    by=["mean_log_loss", "mean_auc"],
    ascending=[True, False]
)

best_C = cv_results_sorted.iloc[0]["C"]

print("Лучшее значение C:", best_C)

final_model = make_logistic_model(C=best_C)

final_model.fit(
    X_train,
    y_train,
    model__sample_weight=w_train_balanced
)

valid_prob = final_model.predict_proba(X_valid)[:, 1]


# In[73]:


plt.figure(figsize=(8, 5))
plt.plot(cv_results["C"], cv_results["mean_log_loss"], marker="o")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Mean weighted Log Loss")
plt.title("Подбор параметра регуляризации C")
plt.grid(True)
plt.show()


# In[74]:


final_model = make_logistic_model(C=best_C)

final_model.fit(
    X_train,
    y_train,
    model__sample_weight=w_train
)

valid_prob = final_model.predict_proba(X_valid)[:, 1]

valid_prob[:10]


# In[75]:


roc_auc = roc_auc_score(
    y_valid,
    valid_prob,
    sample_weight=w_valid
)

pr_auc = average_precision_score(
    y_valid,
    valid_prob,
    sample_weight=w_valid
)

valid_logloss = log_loss(
    y_valid,
    valid_prob,
    sample_weight=w_valid,
    labels=[0, 1]
)

brier = brier_score_loss(
    y_valid,
    valid_prob,
    sample_weight=w_valid
)

metrics_table = pd.DataFrame({
    "Метрика": [
        "ROC-AUC",
        "PR-AUC / Average Precision",
        "Log Loss",
        "Brier Score"
    ],
    "Значение": [
        roc_auc,
        pr_auc,
        valid_logloss,
        brier
    ]
})

metrics_table


# In[76]:


RocCurveDisplay.from_predictions(
    y_valid,
    valid_prob,
    sample_weight=w_valid
)

plt.title("ROC-кривая логистической регрессии")
plt.grid(True)
plt.show()


# In[77]:


PrecisionRecallDisplay.from_predictions(
    y_valid,
    valid_prob,
    sample_weight=w_valid
)

plt.title("Precision-Recall кривая")
plt.grid(True)
plt.show()


# In[78]:


train_prevalence = np.average(y_train, weights=w_train)

print(f"Взвешенная частота cancer=1 в train: {train_prevalence:.6f}")


# In[79]:


threshold = train_prevalence

valid_pred = (valid_prob >= threshold).astype(int)

cm = confusion_matrix(
    y_valid,
    valid_pred,
    sample_weight=w_valid
)

cm_table = pd.DataFrame(
    cm,
    index=["Факт: cancer=0", "Факт: cancer=1"],
    columns=["Прогноз: cancer=0", "Прогноз: cancer=1"]
)

cm_table


# In[80]:


report = classification_report(
    y_valid,
    valid_pred,
    sample_weight=w_valid,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report).T


# In[81]:


def weighted_calibration_table(y_true, y_prob, weights, n_bins=10):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "weight": weights
    })
    
    df["bin"] = pd.qcut(
        df["y_prob"],
        q=n_bins,
        duplicates="drop"
    )
    
    rows = []
    
    for interval, part in df.groupby("bin", observed=True):
        mean_predicted = np.average(
            part["y_prob"],
            weights=part["weight"]
        )
        
        observed_rate = np.average(
            part["y_true"],
            weights=part["weight"]
        )
        
        rows.append({
            "risk_bin": str(interval),
            "weighted_count": part["weight"].sum(),
            "mean_predicted_risk": mean_predicted,
            "observed_cancer_rate": observed_rate
        })
    
    return pd.DataFrame(rows)

calibration_table = weighted_calibration_table(
    y_valid,
    valid_prob,
    w_valid,
    n_bins=10
)

calibration_table


# In[82]:


plt.figure(figsize=(6, 6))

plt.plot(
    calibration_table["mean_predicted_risk"],
    calibration_table["observed_cancer_rate"],
    marker="o",
    label="Модель"
)

plt.plot(
    [0, calibration_table["mean_predicted_risk"].max()],
    [0, calibration_table["mean_predicted_risk"].max()],
    linestyle="--",
    label="Идеальная калибровка"
)

plt.xlabel("Средний предсказанный риск")
plt.ylabel("Фактическая доля cancer=1")
plt.title("Калибровка модели")
plt.legend()
plt.grid(True)
plt.show()


# In[83]:


valid_results = valid_df.copy()
valid_results["predicted_risk"] = valid_prob

valid_results[[
    "menopaus",
    "agegrp",
    "density",
    "race",
    "bmi",
    "nrelbc",
    "brstproc",
    "cancer",
    "count",
    "predicted_risk"
]].head(20)


# In[84]:


risk_deciles = weighted_calibration_table(
    y_valid,
    valid_prob,
    w_valid,
    n_bins=10
)

risk_deciles


# In[85]:


value_labels = {
    "menopaus": {
        0: "premenopausal",
        1: "postmenopausal_or_age_55_plus",
        9: "unknown"
    },
    "agegrp": {
        1: "35-39",
        2: "40-44",
        3: "45-49",
        4: "50-54",
        5: "55-59",
        6: "60-64",
        7: "65-69",
        8: "70-74",
        9: "75-79",
        10: "80-84"
    },
    "density": {
        1: "almost_entirely_fat",
        2: "scattered_fibroglandular",
        3: "heterogeneously_dense",
        4: "extremely_dense",
        9: "unknown"
    },
    "race": {
        1: "white",
        2: "asian_pacific_islander",
        3: "black",
        4: "native_american",
        5: "other_mixed",
        9: "unknown"
    },
    "Hispanic": {
        0: "no",
        1: "yes",
        9: "unknown"
    },
    "bmi": {
        1: "10-24.99",
        2: "25-29.99",
        3: "30-34.99",
        4: "35_or_more",
        9: "unknown"
    },
    "agefirst": {
        0: "age_first_birth_under_30",
        1: "age_first_birth_30_or_more",
        2: "nulliparous",
        9: "unknown"
    },
    "nrelbc": {
        0: "zero",
        1: "one",
        2: "two_or_more",
        9: "unknown"
    },
    "brstproc": {
        0: "no",
        1: "yes",
        9: "unknown"
    },
    "lastmamm": {
        0: "negative",
        1: "false_positive",
        9: "unknown"
    },
    "surgmeno": {
        0: "natural",
        1: "surgical",
        9: "unknown_or_not_menopausal"
    },
    "hrt": {
        0: "no",
        1: "yes",
        9: "unknown_or_not_menopausal"
    }
}


# In[86]:


ohe = final_model.named_steps["preprocess"].named_transformers_["cat"]
model = final_model.named_steps["model"]

encoded_feature_names = final_model.named_steps["preprocess"].get_feature_names_out()
coefficients = model.coef_[0]

coef_table = pd.DataFrame({
    "encoded_feature": encoded_feature_names,
    "coef": coefficients
})

coef_table["odds_ratio"] = np.exp(coef_table["coef"])
coef_table["abs_coef"] = coef_table["coef"].abs()

coef_table.head()


# In[87]:


reference_categories = {}

for variable, categories in zip(features, ohe.categories_):
    reference_categories[variable] = categories[0]

reference_categories


# In[88]:


def parse_encoded_feature(encoded_name):
    name = encoded_name.replace("cat__", "")
    variable, category = name.rsplit("_", 1)
    
    try:
        category = int(category)
    except ValueError:
        pass
    
    reference = reference_categories.get(variable)
    
    category_label = value_labels.get(variable, {}).get(category, str(category))
    reference_label = value_labels.get(variable, {}).get(reference, str(reference))
    
    return pd.Series({
        "variable": variable,
        "category": category,
        "category_label": category_label,
        "reference_category": reference,
        "reference_label": reference_label
    })

parsed_features = coef_table["encoded_feature"].apply(parse_encoded_feature)

interpretation_table = pd.concat(
    [parsed_features, coef_table],
    axis=1
)

interpretation_table = interpretation_table[[
    "variable",
    "category",
    "category_label",
    "reference_category",
    "reference_label",
    "coef",
    "odds_ratio",
    "abs_coef"
]]

interpretation_table.sort_values(
    by="abs_coef",
    ascending=False
).head(30)


# In[89]:


positive_effects = interpretation_table[
    interpretation_table["odds_ratio"] > 1
].sort_values(
    by="odds_ratio",
    ascending=False
)

positive_effects.head(20)


# In[90]:


negative_effects = interpretation_table[
    interpretation_table["odds_ratio"] < 1
].sort_values(
    by="odds_ratio",
    ascending=True
)

negative_effects.head(20)


# In[91]:


top_interpretation = interpretation_table.sort_values(
    by="abs_coef",
    ascending=False
).head(20).copy()

plt.figure(figsize=(10, 7))

plt.barh(
    top_interpretation["variable"] + "=" + top_interpretation["category_label"],
    top_interpretation["coef"]
)

plt.xlabel("Коэффициент логистической регрессии")
plt.title("Наиболее сильные признаки по модулю коэффициента")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# In[92]:


top_or = interpretation_table.sort_values(
    by="abs_coef",
    ascending=False
).head(20).copy()

plt.figure(figsize=(10, 7))

plt.barh(
    top_or["variable"] + "=" + top_or["category_label"],
    top_or["odds_ratio"]
)

plt.axvline(1, linestyle="--")
plt.xlabel("Odds Ratio")
plt.title("Отношение шансов для наиболее значимых категорий")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# In[93]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

glm_train_df = train_df.copy()

for col in features:
    glm_train_df[col] = glm_train_df[col].astype("category")

formula = "cancer ~ " + " + ".join(features)

formula


# In[94]:


glm_model = smf.glm(
    formula=formula,
    data=glm_train_df,
    family=sm.families.Binomial(),
    freq_weights=glm_train_df["count"]
)

glm_result = glm_model.fit(maxiter=200)

print(glm_result.summary())


# In[95]:


params = glm_result.params
conf_intervals = glm_result.conf_int()
p_values = glm_result.pvalues

or_ci_table = pd.DataFrame({
    "coef": params,
    "odds_ratio": np.exp(params),
    "ci_lower": np.exp(conf_intervals[0]),
    "ci_upper": np.exp(conf_intervals[1]),
    "p_value": p_values
})

or_ci_table = or_ci_table.drop(index="Intercept", errors="ignore")

or_ci_table = or_ci_table.sort_values(
    by="odds_ratio",
    ascending=False
)

or_ci_table.head(30)


# In[96]:


tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

classification_metrics = pd.DataFrame({
    "Метрика": [
        "Accuracy",
        "Precision",
        "Recall / Sensitivity",
        "Specificity",
        "F1-score"
    ],
    "Значение": [
        accuracy,
        precision,
        recall,
        specificity,
        f1
    ]
})

classification_metrics


# In[97]:


plt.figure(figsize=(9, 5))

plt.bar(
    classification_metrics["Метрика"],
    classification_metrics["Значение"]
)

plt.ylim(0, 1)
plt.ylabel("Значение")
plt.title("Метрики качества классификации")

for i, value in enumerate(classification_metrics["Значение"]):
    plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# In[98]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve
)


# In[99]:


try:
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse=True
    )

preprocessor_svm = ColumnTransformer(
    transformers=[
        ("cat", encoder, features)
    ],
    remainder="drop"
)

def make_svm_model(C=1.0):
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor_svm),
            ("svm", LinearSVC(
                C=C,
                penalty="l2",
                loss="squared_hinge",
                dual=False,
                max_iter=10000,
                random_state=42
            ))
        ]
    )
    
    return model


# In[100]:


svm_baseline = make_svm_model(C=1.0)

svm_baseline.fit(
    X_train,
    y_train,
    svm__sample_weight=w_train_balanced
)

valid_scores_baseline = svm_baseline.decision_function(X_valid)

baseline_auc = roc_auc_score(
    y_valid,
    valid_scores_baseline,
    sample_weight=w_valid
)

baseline_pr_auc = average_precision_score(
    y_valid,
    valid_scores_baseline,
    sample_weight=w_valid
)

print(f"Baseline SVM ROC-AUC: {baseline_auc:.4f}")
print(f"Baseline SVM PR-AUC: {baseline_pr_auc:.4f}")


# In[101]:


C_grid = np.logspace(-3, 2, 6)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

svm_cv_results = []

for C in C_grid:
    fold_auc = []
    fold_pr_auc = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        w_tr = w_train.iloc[train_idx]
        
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        w_val = w_train.iloc[val_idx]
        
        w_tr_balanced = make_balanced_weights(y_tr, w_tr)
        
        model = make_svm_model(C=C)
        
        model.fit(
            X_tr,
            y_tr,
            svm__sample_weight=w_tr_balanced
        )
        
        val_scores = model.decision_function(X_val)
        
        fold_auc.append(
            roc_auc_score(
                y_val,
                val_scores,
                sample_weight=w_val
            )
        )
        
        fold_pr_auc.append(
            average_precision_score(
                y_val,
                val_scores,
                sample_weight=w_val
            )
        )
    
    svm_cv_results.append({
        "C": C,
        "mean_roc_auc": np.mean(fold_auc),
        "std_roc_auc": np.std(fold_auc),
        "mean_pr_auc": np.mean(fold_pr_auc),
        "std_pr_auc": np.std(fold_pr_auc)
    })

svm_cv_results = pd.DataFrame(svm_cv_results)

svm_cv_results


# In[102]:


svm_cv_results_sorted = svm_cv_results.sort_values(
    by=["mean_roc_auc", "mean_pr_auc"],
    ascending=[False, False]
)

best_svm_C = svm_cv_results_sorted.iloc[0]["C"]

print("Лучшее значение C для SVM:", best_svm_C)

final_svm = make_svm_model(C=best_svm_C)

final_svm.fit(
    X_train,
    y_train,
    svm__sample_weight=w_train_balanced
)

train_scores = final_svm.decision_function(X_train)
valid_scores = final_svm.decision_function(X_valid)


# In[103]:


plt.figure(figsize=(8, 5))
plt.plot(
    svm_cv_results["C"],
    svm_cv_results["mean_roc_auc"],
    marker="o"
)

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Mean weighted ROC-AUC")
plt.title("Подбор параметра C для линейного SVM")
plt.grid(True)
plt.show()


# In[104]:


final_svm = make_svm_model(C=best_svm_C)

final_svm.fit(
    X_train,
    y_train,
    svm__sample_weight=w_train
)

train_scores = final_svm.decision_function(X_train)
valid_scores = final_svm.decision_function(X_valid)

valid_scores[:10]


# In[105]:


svm_roc_auc = roc_auc_score(
    y_valid,
    valid_scores,
    sample_weight=w_valid
)

svm_pr_auc = average_precision_score(
    y_valid,
    valid_scores,
    sample_weight=w_valid
)

svm_metrics_table = pd.DataFrame({
    "Метрика": [
        "ROC-AUC",
        "PR-AUC / Average Precision"
    ],
    "Значение": [
        svm_roc_auc,
        svm_pr_auc
    ]
})

svm_metrics_table


# In[106]:


RocCurveDisplay.from_predictions(
    y_valid,
    valid_scores,
    sample_weight=w_valid
)

plt.title("ROC-кривая модели SVM")
plt.grid(True)
plt.show()


# In[107]:


PrecisionRecallDisplay.from_predictions(
    y_valid,
    valid_scores,
    sample_weight=w_valid
)

plt.title("Precision-Recall кривая модели SVM")
plt.grid(True)
plt.show()


# In[108]:


default_threshold = 0

valid_pred_default = (valid_scores >= default_threshold).astype(int)

cm_default = confusion_matrix(
    y_valid,
    valid_pred_default,
    sample_weight=w_valid
)

cm_default_table = pd.DataFrame(
    cm_default,
    index=["Факт: cancer=0", "Факт: cancer=1"],
    columns=["Прогноз: cancer=0", "Прогноз: cancer=1"]
)

cm_default_table


# In[109]:


report_default = classification_report(
    y_valid,
    valid_pred_default,
    sample_weight=w_valid,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report_default).T


# In[110]:


fpr, tpr, thresholds = roc_curve(
    y_train,
    train_scores,
    sample_weight=w_train
)

youden_j = tpr - fpr
best_threshold_index = np.argmax(youden_j)
best_svm_threshold = thresholds[best_threshold_index]

print("Оптимальный порог SVM по Youden's J:", best_svm_threshold)
print("TPR на train:", tpr[best_threshold_index])
print("FPR на train:", fpr[best_threshold_index])


# In[111]:


valid_pred_youden = (valid_scores >= best_svm_threshold).astype(int)

cm_youden = confusion_matrix(
    y_valid,
    valid_pred_youden,
    sample_weight=w_valid
)

cm_youden_table = pd.DataFrame(
    cm_youden,
    index=["Факт: cancer=0", "Факт: cancer=1"],
    columns=["Прогноз: cancer=0", "Прогноз: cancer=1"]
)

cm_youden_table


# In[112]:


report_youden = classification_report(
    y_valid,
    valid_pred_youden,
    sample_weight=w_valid,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report_youden).T


# In[113]:


def weighted_classification_metrics(y_true, y_pred, weights):
    cm = confusion_matrix(
        y_true,
        y_pred,
        sample_weight=weights
    )
    
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1
    }

default_metrics = weighted_classification_metrics(
    y_valid,
    valid_pred_default,
    w_valid
)

youden_metrics = weighted_classification_metrics(
    y_valid,
    valid_pred_youden,
    w_valid
)

threshold_comparison = pd.DataFrame([
    {
        "threshold_type": "default_0",
        "threshold": default_threshold,
        **default_metrics
    },
    {
        "threshold_type": "youden_train",
        "threshold": best_svm_threshold,
        **youden_metrics
    }
])

threshold_comparison


# In[114]:


scores_df = pd.DataFrame({
    "score": valid_scores,
    "cancer": y_valid,
    "count": w_valid
})

plt.figure(figsize=(8, 5))

plt.hist(
    scores_df.loc[scores_df["cancer"] == 0, "score"],
    bins=50,
    alpha=0.6,
    weights=scores_df.loc[scores_df["cancer"] == 0, "count"],
    label="cancer=0"
)

plt.hist(
    scores_df.loc[scores_df["cancer"] == 1, "score"],
    bins=50,
    alpha=0.6,
    weights=scores_df.loc[scores_df["cancer"] == 1, "count"],
    label="cancer=1"
)

plt.axvline(default_threshold, linestyle="--", label="threshold=0")
plt.axvline(best_svm_threshold, linestyle="--", label="Youden threshold")

plt.xlabel("Значение решающей функции SVM")
plt.ylabel("Взвешенная частота")
plt.title("Распределение SVM-score по классам")
plt.legend()
plt.grid(True)
plt.show()


# In[115]:


svm_final_results = pd.DataFrame({
    "Показатель": [
        "ROC-AUC",
        "PR-AUC",
        "Порог по умолчанию",
        "Порог по Youden's J",
        "Accuracy при пороге 0",
        "Recall при пороге 0",
        "Precision при пороге 0",
        "F1-score при пороге 0",
        "Accuracy при пороге Youden",
        "Recall при пороге Youden",
        "Precision при пороге Youden",
        "F1-score при пороге Youden"
    ],
    "Значение": [
        svm_roc_auc,
        svm_pr_auc,
        default_threshold,
        best_svm_threshold,
        default_metrics["accuracy"],
        default_metrics["sensitivity_recall"],
        default_metrics["precision"],
        default_metrics["f1_score"],
        youden_metrics["accuracy"],
        youden_metrics["sensitivity_recall"],
        youden_metrics["precision"],
        youden_metrics["f1_score"]
    ]
})

svm_final_results


# In[ ]:





# In[116]:


import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)


# In[117]:


def make_balanced_weights(y, weights, normalize=True):
    y = pd.Series(y).reset_index(drop=True)
    weights = pd.Series(weights).reset_index(drop=True).astype(float)
    
    class_0_sum = weights[y == 0].sum()
    class_1_sum = weights[y == 1].sum()
    total_sum = weights.sum()
    
    class_weight_0 = total_sum / (2 * class_0_sum)
    class_weight_1 = total_sum / (2 * class_1_sum)
    
    balanced_weights = weights.copy()
    balanced_weights[y == 0] = weights[y == 0] * class_weight_0
    balanced_weights[y == 1] = weights[y == 1] * class_weight_1
    
    if normalize:
        balanced_weights = balanced_weights / balanced_weights.mean()
    
    return balanced_weights

w_train_nn = make_balanced_weights(
    y_train,
    w_train,
    normalize=True
)

w_valid_nn = w_valid.astype(float) / w_valid.astype(float).mean()

print("Средний вес train после балансировки:", w_train_nn.mean())
print("Средний вес validation:", w_valid_nn.mean())

print("Суммарный вес класса 0 train:", w_train_nn[y_train.reset_index(drop=True) == 0].sum())
print("Суммарный вес класса 1 train:", w_train_nn[y_train.reset_index(drop=True) == 1].sum())


# In[118]:


try:
    encoder_nn = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder_nn = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

X_train_ohe = encoder_nn.fit_transform(X_train).astype("float32")
X_valid_ohe = encoder_nn.transform(X_valid).astype("float32")

y_train_np = y_train.to_numpy().astype("float32")
y_valid_np = y_valid.to_numpy().astype("float32")

w_train_nn_np = w_train_nn.to_numpy().astype("float32")
w_valid_nn_np = w_valid_nn.to_numpy().astype("float32")

print("X_train_ohe:", X_train_ohe.shape)
print("X_valid_ohe:", X_valid_ohe.shape)


# In[119]:


def build_token_mapping(dataframe, features):
    token_to_id = {}
    token_to_id["UNKNOWN"] = 0
    
    current_id = 1
    
    for feature in features:
        values = sorted(dataframe[feature].unique())
        
        for value in values:
            token = f"{feature}={value}"
            
            if token not in token_to_id:
                token_to_id[token] = current_id
                current_id += 1
    
    return token_to_id

def dataframe_to_token_sequence(dataframe, features, token_to_id):
    sequences = []
    
    for _, row in dataframe[features].iterrows():
        sequence = []
        
        for feature in features:
            token = f"{feature}={row[feature]}"
            token_id = token_to_id.get(token, 0)
            sequence.append(token_id)
        
        sequences.append(sequence)
    
    return np.array(sequences, dtype="int32")

token_to_id = build_token_mapping(train_df, features)

X_train_seq = dataframe_to_token_sequence(
    train_df,
    features,
    token_to_id
)

X_valid_seq = dataframe_to_token_sequence(
    valid_df,
    features,
    token_to_id
)

vocab_size = len(token_to_id)
seq_len = len(features)

print("Количество токенов:", vocab_size)
print("Длина последовательности:", seq_len)

print("X_train_seq:", X_train_seq.shape)
print("X_valid_seq:", X_valid_seq.shape)


# In[120]:


def build_mlp_simple(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="MLP_simple")
    
    return model


# In[121]:


def build_mlp_deep(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(32, activation="relu")(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="MLP_deep")
    
    return model


# In[122]:


def build_mlp_wide(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="MLP_wide")
    
    return model


# In[123]:


def residual_block(x, units, dropout_rate=0.25):
    shortcut = x
    
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units, activation=None)(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != units:
        shortcut = layers.Dense(units, activation=None)(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    
    return x

def build_mlp_residual(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    
    x = residual_block(x, 128, dropout_rate=0.30)
    x = residual_block(x, 128, dropout_rate=0.30)
    x = residual_block(x, 64, dropout_rate=0.25)
    
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="MLP_residual")
    
    return model


# In[124]:


def build_embedding_mlp(vocab_size, seq_len, embedding_dim=16):
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )(inputs)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="Embedding_MLP")
    
    return model


# In[125]:


def build_lstm_model(vocab_size, seq_len, embedding_dim=16):
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )(inputs)
    
    x = layers.LSTM(
        64,
        return_sequences=False
    )(x)
    
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="Embedding_LSTM")
    
    return model


# In[126]:


def build_gru_model(vocab_size, seq_len, embedding_dim=16):
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )(inputs)
    
    x = layers.GRU(
        64,
        return_sequences=False
    )(x)
    
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="Embedding_GRU")
    
    return model


# In[127]:


def build_cnn1d_model(vocab_size, seq_len, embedding_dim=16):
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )(inputs)
    
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)
    
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)
    
    x = layers.GlobalMaxPooling1D()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="Embedding_CNN1D")
    
    return model


# In[128]:


def build_attention_model(vocab_size, seq_len, embedding_dim=16):
    inputs = layers.Input(shape=(seq_len,), dtype="int32")
    
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )(inputs)
    
    attention_output = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=embedding_dim
    )(x, x)
    
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    x_ff = layers.Dense(64, activation="relu")(x)
    x_ff = layers.Dense(embedding_dim)(x_ff)
    
    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="Embedding_Attention")
    
    return model


# In[129]:


def compile_neural_model(model, learning_rate=0.001):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )
    
    return model


# In[130]:


neural_architectures = [
    {
        "name": "MLP_simple",
        "input_type": "onehot",
        "builder": lambda: build_mlp_simple(X_train_ohe.shape[1])
    },
    {
        "name": "MLP_deep",
        "input_type": "onehot",
        "builder": lambda: build_mlp_deep(X_train_ohe.shape[1])
    },
    {
        "name": "MLP_wide",
        "input_type": "onehot",
        "builder": lambda: build_mlp_wide(X_train_ohe.shape[1])
    },
    {
        "name": "MLP_residual",
        "input_type": "onehot",
        "builder": lambda: build_mlp_residual(X_train_ohe.shape[1])
    },
    {
        "name": "Embedding_MLP",
        "input_type": "sequence",
        "builder": lambda: build_embedding_mlp(vocab_size, seq_len)
    },
    {
        "name": "Embedding_LSTM",
        "input_type": "sequence",
        "builder": lambda: build_lstm_model(vocab_size, seq_len)
    },
    {
        "name": "Embedding_GRU",
        "input_type": "sequence",
        "builder": lambda: build_gru_model(vocab_size, seq_len)
    },
    {
        "name": "Embedding_CNN1D",
        "input_type": "sequence",
        "builder": lambda: build_cnn1d_model(vocab_size, seq_len)
    },
    {
        "name": "Embedding_Attention",
        "input_type": "sequence",
        "builder": lambda: build_attention_model(vocab_size, seq_len)
    }
]


# In[131]:


EPOCHS = 50
BATCH_SIZE = 4096

trained_neural_models = {}
neural_histories = {}
neural_results = []

for architecture in neural_architectures:
    tf.keras.backend.clear_session()
    
    model_name = architecture["name"]
    input_type = architecture["input_type"]
    
    print("=" * 80)
    print("Обучается модель:", model_name)
    print("=" * 80)
    
    model = architecture["builder"]()
    model = compile_neural_model(model, learning_rate=0.001)
    
    if input_type == "onehot":
        X_tr = X_train_ohe
        X_val = X_valid_ohe
    else:
        X_tr = X_train_seq
        X_val = X_valid_seq
    
    early_stop = callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=6,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )
    
    history = model.fit(
        X_tr,
        y_train_np,
        sample_weight=w_train_nn_np,
        validation_data=(X_val, y_valid_np, w_valid_nn_np),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    valid_prob = model.predict(
        X_val,
        batch_size=BATCH_SIZE
    ).ravel()
    
    valid_prob = np.clip(valid_prob, 1e-7, 1 - 1e-7)
    
    roc_auc = roc_auc_score(
        y_valid,
        valid_prob,
        sample_weight=w_valid
    )
    
    pr_auc = average_precision_score(
        y_valid,
        valid_prob,
        sample_weight=w_valid
    )
    
    valid_logloss = log_loss(
        y_valid,
        valid_prob,
        sample_weight=w_valid,
        labels=[0, 1]
    )
    
    brier = brier_score_loss(
        y_valid,
        valid_prob,
        sample_weight=w_valid
    )
    
    neural_results.append({
        "model": model_name,
        "input_type": input_type,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": valid_logloss,
        "brier_score": brier,
        "epochs_trained": len(history.history["loss"])
    })
    
    trained_neural_models[model_name] = {
        "model": model,
        "input_type": input_type
    }
    
    neural_histories[model_name] = history
    
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("Log Loss:", valid_logloss)
    print("Brier Score:", brier)


# In[132]:


neural_results_df = pd.DataFrame(neural_results)

neural_results_df = neural_results_df.sort_values(
    by=["pr_auc", "roc_auc"],
    ascending=[False, False]
).reset_index(drop=True)

neural_results_df


# In[133]:


plt.figure(figsize=(10, 5))

plt.bar(
    neural_results_df["model"],
    neural_results_df["roc_auc"]
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC")
plt.title("Сравнение нейросетевых моделей по ROC-AUC")
plt.grid(True)
plt.show()


# In[134]:


plt.figure(figsize=(10, 5))

plt.bar(
    neural_results_df["model"],
    neural_results_df["pr_auc"]
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("PR-AUC")
plt.title("Сравнение нейросетевых моделей по PR-AUC")
plt.grid(True)
plt.show()


# In[135]:


best_neural_model_name = neural_results_df.iloc[0]["model"]

best_neural_model_info = trained_neural_models[best_neural_model_name]
best_neural_model = best_neural_model_info["model"]
best_input_type = best_neural_model_info["input_type"]

print("Лучшая нейросетевая модель:", best_neural_model_name)
print("Тип входных данных:", best_input_type)

if best_input_type == "onehot":
    X_best_valid = X_valid_ohe
    X_best_train = X_train_ohe
else:
    X_best_valid = X_valid_seq
    X_best_train = X_train_seq

best_valid_prob = best_neural_model.predict(
    X_best_valid,
    batch_size=BATCH_SIZE
).ravel()

best_train_prob = best_neural_model.predict(
    X_best_train,
    batch_size=BATCH_SIZE
).ravel()

best_valid_prob = np.clip(best_valid_prob, 1e-7, 1 - 1e-7)
best_train_prob = np.clip(best_train_prob, 1e-7, 1 - 1e-7)


# In[136]:


RocCurveDisplay.from_predictions(
    y_valid,
    best_valid_prob,
    sample_weight=w_valid
)

plt.title(f"ROC-кривая лучшей нейросети: {best_neural_model_name}")
plt.grid(True)
plt.show()


# In[137]:


fpr, tpr, thresholds = roc_curve(
    y_train,
    best_train_prob,
    sample_weight=w_train
)

youden_j = tpr - fpr
best_threshold_index = np.argmax(youden_j)
best_nn_threshold = thresholds[best_threshold_index]

print("Оптимальный порог по Youden's J:", best_nn_threshold)
print("TPR на train:", tpr[best_threshold_index])
print("FPR на train:", fpr[best_threshold_index])


# In[138]:


valid_pred_05 = (best_valid_prob >= 0.5).astype(int)

cm_05 = confusion_matrix(
    y_valid,
    valid_pred_05,
    sample_weight=w_valid
)

cm_05_table = pd.DataFrame(
    cm_05,
    index=["Факт: cancer=0", "Факт: cancer=1"],
    columns=["Прогноз: cancer=0", "Прогноз: cancer=1"]
)

cm_05_table


# In[139]:


best_history = neural_histories[best_neural_model_name]

plt.figure(figsize=(8, 5))

plt.plot(
    best_history.history["loss"],
    label="Train loss"
)

plt.plot(
    best_history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.title(f"Динамика ошибки: {best_neural_model_name}")
plt.legend()
plt.grid(True)
plt.show()


# In[140]:


plt.figure(figsize=(8, 5))

plt.plot(
    best_history.history["pr_auc"],
    label="Train PR-AUC"
)

plt.plot(
    best_history.history["val_pr_auc"],
    label="Validation PR-AUC"
)

plt.xlabel("Эпоха")
plt.ylabel("PR-AUC")
plt.title(f"Динамика PR-AUC: {best_neural_model_name}")
plt.legend()
plt.grid(True)
plt.show()


# In[141]:


best_nn_metrics = neural_results_df.iloc[0]

best_nn_summary = pd.DataFrame({
    "model_type": ["Neural network"],
    "model_name": [best_nn_metrics["model"]],
    "roc_auc": [best_nn_metrics["roc_auc"]],
    "pr_auc": [best_nn_metrics["pr_auc"]],
    "log_loss": [best_nn_metrics["log_loss"]],
    "brier_score": [best_nn_metrics["brier_score"]]
})

best_nn_summary


# In[142]:


logistic_summary = pd.DataFrame({
    "model_type": ["Logistic regression"],
    "model_name": ["Logistic regression"],
    "roc_auc": [roc_auc],
    "pr_auc": [pr_auc],
    "log_loss": [valid_logloss],
    "brier_score": [brier]
})

logistic_summary


# In[143]:


svm_summary = pd.DataFrame({
    "model_type": ["SVM"],
    "model_name": ["LinearSVM"],
    "roc_auc": [svm_roc_auc],
    "pr_auc": [svm_pr_auc],
    "log_loss": [np.nan],
    "brier_score": [np.nan]
})

svm_summary


# In[144]:


model_summaries = []

if "logistic_summary" in globals():
    model_summaries.append(logistic_summary)

if "svm_summary" in globals():
    model_summaries.append(svm_summary)

model_summaries.append(best_nn_summary)

final_model_comparison = pd.concat(
    model_summaries,
    ignore_index=True
)

final_model_comparison = final_model_comparison.sort_values(
    by=["pr_auc", "roc_auc"],
    ascending=[False, False]
).reset_index(drop=True)

final_model_comparison


# In[ ]:





# In[ ]:





# In[145]:


import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)


# In[146]:


def make_balanced_weights(y, weights, normalize=True):
    y = pd.Series(y).reset_index(drop=True)
    weights = pd.Series(weights).reset_index(drop=True).astype(float)
    
    class_0_sum = weights[y == 0].sum()
    class_1_sum = weights[y == 1].sum()
    total_sum = weights.sum()
    
    class_weight_0 = total_sum / (2 * class_0_sum)
    class_weight_1 = total_sum / (2 * class_1_sum)
    
    balanced_weights = weights.copy()
    balanced_weights[y == 0] = weights[y == 0] * class_weight_0
    balanced_weights[y == 1] = weights[y == 1] * class_weight_1
    
    if normalize:
        balanced_weights = balanced_weights / balanced_weights.mean()
    
    return balanced_weights

w_train_clf = make_balanced_weights(
    y_train,
    w_train,
    normalize=True
)

print("Средний вес после нормализации:", w_train_clf.mean())

print("Суммарный вес класса 0 после балансировки:")
print(w_train_clf[y_train.reset_index(drop=True) == 0].sum())

print("Суммарный вес класса 1 после балансировки:")
print(w_train_clf[y_train.reset_index(drop=True) == 1].sum())

try:
    encoder_clf = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder_clf = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

X_train_clf = encoder_clf.fit_transform(X_train).astype("float32")
X_valid_clf = encoder_clf.transform(X_valid).astype("float32")

y_train_clf = y_train.to_numpy()
y_valid_clf = y_valid.to_numpy()

w_train_clf_np = w_train_clf.to_numpy().astype("float32")
w_valid_np = w_valid.to_numpy().astype("float32")

print("X_train_clf:", X_train_clf.shape)
print("X_valid_clf:", X_valid_clf.shape)


# In[ ]:





# In[ ]:





# In[147]:


RUN_SLOW_MODELS = True

classification_models = {
    "LogisticRegression": LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        random_state=42
    ),
    
    "SGD_LogLoss": SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=2000,
        tol=1e-4,
        random_state=42
    ),
    
    "RidgeClassifier": RidgeClassifier(
        alpha=1.0,
        random_state=42
    ),
    
    "LinearSVC": LinearSVC(
        C=1.0,
        penalty="l2",
        loss="squared_hinge",
        dual=False,
        max_iter=10000,
        random_state=42
    ),
    
    "BernoulliNB": BernoulliNB(
        alpha=1.0
    ),
    
    "DecisionTree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=100,
        random_state=42
    ),
    
    "RandomForest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42
    ),
    
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42
    ),
    
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42
    )
}

if RUN_SLOW_MODELS:
    classification_models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=100,
        random_state=42
    )
    
    classification_models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=42
    )

def get_model_scores(model, X):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        score_type = "probability"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        score_type = "decision_function"
    else:
        scores = model.predict(X)
        score_type = "class_prediction"
    
    return scores, score_type

def get_default_threshold(score_type):
    if score_type == "probability":
        return 0.5
    else:
        return 0.0

def find_youden_threshold(y_true, scores, weights):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        sample_weight=weights
    )
    
    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)
    
    return thresholds[best_index], tpr[best_index], fpr[best_index]

def weighted_classification_metrics(y_true, y_pred, weights):
    cm = confusion_matrix(
        y_true,
        y_pred,
        sample_weight=weights
    )
    
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    
    return {
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1
    }


# In[148]:


def evaluate_classifier(
    model_name,
    model,
    X_train_data,
    y_train_data,
    w_train_data,
    X_valid_data,
    y_valid_data,
    w_valid_data
):
    start_time = time.time()
    
    model.fit(
        X_train_data,
        y_train_data,
        sample_weight=w_train_data
    )
    
    fit_time = time.time() - start_time
    
    train_scores, train_score_type = get_model_scores(
        model,
        X_train_data
    )
    
    valid_scores, valid_score_type = get_model_scores(
        model,
        X_valid_data
    )
    
    roc_auc = roc_auc_score(
        y_valid_data,
        valid_scores,
        sample_weight=w_valid_data
    )
    
    pr_auc = average_precision_score(
        y_valid_data,
        valid_scores,
        sample_weight=w_valid_data
    )
    
    if valid_score_type == "probability":
        valid_prob = np.clip(valid_scores, 1e-7, 1 - 1e-7)
        
        model_log_loss = log_loss(
            y_valid_data,
            valid_prob,
            sample_weight=w_valid_data,
            labels=[0, 1]
        )
        
        model_brier = brier_score_loss(
            y_valid_data,
            valid_prob,
            sample_weight=w_valid_data
        )
    else:
        model_log_loss = np.nan
        model_brier = np.nan
    
    default_threshold = get_default_threshold(valid_score_type)
    
    valid_pred_default = (valid_scores >= default_threshold).astype(int)
    
    default_metrics = weighted_classification_metrics(
        y_valid_data,
        valid_pred_default,
        w_valid_data
    )
    
    youden_threshold, train_tpr, train_fpr = find_youden_threshold(
        y_train_data,
        train_scores,
        w_train_data
    )
    
    valid_pred_youden = (valid_scores >= youden_threshold).astype(int)
    
    youden_metrics = weighted_classification_metrics(
        y_valid_data,
        valid_pred_youden,
        w_valid_data
    )
    
    result = {
        "model": model_name,
        "score_type": valid_score_type,
        "fit_time_sec": fit_time,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": model_log_loss,
        "brier_score": model_brier,
        "default_threshold": default_threshold,
        "youden_threshold": youden_threshold,
        "default_accuracy": default_metrics["accuracy"],
        "default_recall": default_metrics["recall"],
        "default_specificity": default_metrics["specificity"],
        "default_precision": default_metrics["precision"],
        "default_f1": default_metrics["f1_score"],
        "youden_accuracy": youden_metrics["accuracy"],
        "youden_recall": youden_metrics["recall"],
        "youden_specificity": youden_metrics["specificity"],
        "youden_precision": youden_metrics["precision"],
        "youden_f1": youden_metrics["f1_score"]
    }
    
    prediction_info = {
        "model": model,
        "valid_scores": valid_scores,
        "train_scores": train_scores,
        "score_type": valid_score_type,
        "default_threshold": default_threshold,
        "youden_threshold": youden_threshold
    }
    
    return result, prediction_info


# In[149]:


classification_results = []
classification_predictions = {}

for model_name, model in classification_models.items():
    print("=" * 90)
    print("Обучается модель:", model_name)
    print("=" * 90)
    
    try:
        result, prediction_info = evaluate_classifier(
            model_name=model_name,
            model=model,
            X_train_data=X_train_clf,
            y_train_data=y_train_clf,
            w_train_data=w_train_clf_np,
            X_valid_data=X_valid_clf,
            y_valid_data=y_valid_clf,
            w_valid_data=w_valid_np
        )
        
        classification_results.append(result)
        classification_predictions[model_name] = prediction_info
        
        print("ROC-AUC:", result["roc_auc"])
        print("PR-AUC:", result["pr_auc"])
        print("Recall по Youden:", result["youden_recall"])
        print("Precision по Youden:", result["youden_precision"])
        print("F1 по Youden:", result["youden_f1"])
        print("Время обучения, сек:", result["fit_time_sec"])
    
    except Exception as error:
        print("Ошибка при обучении модели:", model_name)
        print(error)


# In[150]:


classification_results = []
classification_predictions = {}

for model_name, model in classification_models.items():
    print("=" * 90)
    print("Обучается модель:", model_name)
    print("=" * 90)
    
    try:
        result, prediction_info = evaluate_classifier(
            model_name=model_name,
            model=model,
            X_train_data=X_train_clf,
            y_train_data=y_train_clf,
            w_train_data=w_train_clf_np,
            X_valid_data=X_valid_clf,
            y_valid_data=y_valid_clf,
            w_valid_data=w_valid_np
        )
        
        classification_results.append(result)
        classification_predictions[model_name] = prediction_info
        
        print("ROC-AUC:", result["roc_auc"])
        print("PR-AUC:", result["pr_auc"])
        print("Recall по Youden:", result["youden_recall"])
        print("Precision по Youden:", result["youden_precision"])
        print("F1 по Youden:", result["youden_f1"])
        print("Время обучения, сек:", result["fit_time_sec"])
    
    except Exception as error:
        print("Ошибка при обучении модели:", model_name)
        print(error)


# In[151]:


classification_results_df = pd.DataFrame(classification_results)

classification_results_df = classification_results_df.sort_values(
    by=["pr_auc", "roc_auc"],
    ascending=[False, False]
).reset_index(drop=True)

classification_results_df


# In[152]:


plt.figure(figsize=(11, 5))

plt.bar(
    classification_results_df["model"],
    classification_results_df["roc_auc"]
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC")
plt.title("Сравнение моделей классификации по ROC-AUC")
plt.grid(True)
plt.show()


# In[153]:


import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost доступен")
except Exception as error:
    XGBOOST_AVAILABLE = False
    print("XGBoost недоступен:", error)

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
    print("imbalanced-learn доступен")
except Exception as error:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn недоступен:", error)


# In[154]:


try:
    encoder_fusion = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder_fusion = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

X_train_ohe = encoder_fusion.fit_transform(X_train)
X_valid_ohe = encoder_fusion.transform(X_valid)

y_train_np = y_train.to_numpy().astype(int)
y_valid_np = y_valid.to_numpy().astype(int)

w_train_np = w_train.to_numpy().astype(float)
w_valid_np = w_valid.to_numpy().astype(float)

print("X_train_ohe:", X_train_ohe.shape)
print("X_valid_ohe:", X_valid_ohe.shape)


# In[155]:


splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

inner_train_idx, inner_tune_idx = next(
    splitter.split(X_train_ohe, y_train_np)
)

X_inner_train = X_train_ohe[inner_train_idx]
y_inner_train = y_train_np[inner_train_idx]
w_inner_train = w_train_np[inner_train_idx]

X_inner_tune = X_train_ohe[inner_tune_idx]
y_inner_tune = y_train_np[inner_tune_idx]
w_inner_tune = w_train_np[inner_tune_idx]

print("Inner train:", X_inner_train.shape)
print("Inner tune:", X_inner_tune.shape)

print("Inner train weighted cancer rate:", np.average(y_inner_train, weights=w_inner_train))
print("Inner tune weighted cancer rate:", np.average(y_inner_tune, weights=w_inner_tune))


# In[157]:


def make_balanced_weights(y, weights, normalize=True):
    y = np.asarray(y)
    weights = np.asarray(weights).astype(float)

    class_0_weight_sum = weights[y == 0].sum()
    class_1_weight_sum = weights[y == 1].sum()
    total_weight_sum = weights.sum()

    class_weight_0 = total_weight_sum / (2 * class_0_weight_sum)
    class_weight_1 = total_weight_sum / (2 * class_1_weight_sum)

    balanced_weights = weights.copy()

    balanced_weights[y == 0] = weights[y == 0] * class_weight_0
    balanced_weights[y == 1] = weights[y == 1] * class_weight_1

    if normalize:
        balanced_weights = balanced_weights / balanced_weights.mean()

    return balanced_weights

w_inner_train_balanced = make_balanced_weights(
    y_inner_train,
    w_inner_train,
    normalize=True
)

w_train_balanced_full = make_balanced_weights(
    y_train_np,
    w_train_np,
    normalize=True
)

print("До балансировки:")
print("class 0 weight:", w_inner_train[y_inner_train == 0].sum())
print("class 1 weight:", w_inner_train[y_inner_train == 1].sum())

print("\nПосле балансировки:")
print("class 0 weight:", w_inner_train_balanced[y_inner_train == 0].sum())
print("class 1 weight:", w_inner_train_balanced[y_inner_train == 1].sum())


# In[159]:


def find_best_f1_threshold(y_true, y_prob, weights):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob,
        sample_weight=weights
    )

    f1_values = []

    for p, r in zip(precision[:-1], recall[:-1]):
        if p + r == 0:
            f1_values.append(0)
        else:
            f1_values.append(2 * p * r / (p + r))

    f1_values = np.array(f1_values)

    best_index = np.argmax(f1_values)
    best_threshold = thresholds[best_index]

    return best_threshold, precision[best_index], recall[best_index], f1_values[best_index]

def find_youden_threshold(y_true, y_prob, weights):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob,
        sample_weight=weights
    )

    youden_j = tpr - fpr
    best_index = np.argmax(youden_j)

    return thresholds[best_index], tpr[best_index], fpr[best_index], youden_j[best_index]


# In[160]:


def evaluate_probabilistic_model(
    model_name,
    y_true,
    y_prob,
    weights,
    threshold
):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "model": model_name,
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
            sample_weight=weights
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
            sample_weight=weights
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            sample_weight=weights,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            sample_weight=weights,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            sample_weight=weights,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
            sample_weight=weights
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob,
            sample_weight=weights
        ),

        "log_loss": log_loss(
            y_true,
            y_prob,
            sample_weight=weights,
            labels=[0, 1]
        ),

        "brier_score": brier_score_loss(
            y_true,
            y_prob,
            sample_weight=weights
        )
    }

    return result


# In[161]:


def get_confusion_matrix_table(y_true, y_prob, weights, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        sample_weight=weights
    )

    cm_table = pd.DataFrame(
        cm,
        index=["Факт: cancer=0", "Факт: cancer=1"],
        columns=["Прогноз: cancer=0", "Прогноз: cancer=1"]
    )

    return cm_table


# In[162]:


def make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.1
):
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    return model


# In[163]:


if not XGBOOST_AVAILABLE:
    raise ImportError("Установи xgboost: !pip install xgboost")

xgb_balanced = make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.1
)

start_time = time.time()

xgb_balanced.fit(
    X_inner_train,
    y_inner_train,
    sample_weight=w_inner_train_balanced
)

fit_time = time.time() - start_time

tune_prob_balanced = xgb_balanced.predict_proba(X_inner_tune)[:, 1]

best_threshold_f1, tune_precision, tune_recall, tune_f1 = find_best_f1_threshold(
    y_inner_tune,
    tune_prob_balanced,
    w_inner_tune
)

print("Время обучения:", fit_time)
print("Лучший порог по F1:", best_threshold_f1)
print("Tune precision:", tune_precision)
print("Tune recall:", tune_recall)
print("Tune F1:", tune_f1)


# In[164]:


xgb_balanced_final = make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.1
)

xgb_balanced_final.fit(
    X_train_ohe,
    y_train_np,
    sample_weight=w_train_balanced_full
)

valid_prob_xgb_balanced = xgb_balanced_final.predict_proba(X_valid_ohe)[:, 1]

result_xgb_balanced = evaluate_probabilistic_model(
    model_name="BalancedWeight_XGBoost",
    y_true=y_valid_np,
    y_prob=valid_prob_xgb_balanced,
    weights=w_valid_np,
    threshold=best_threshold_f1
)

result_xgb_balanced


# In[165]:


cm_xgb_balanced = get_confusion_matrix_table(
    y_valid_np,
    valid_prob_xgb_balanced,
    w_valid_np,
    best_threshold_f1
)

cm_xgb_balanced


# In[166]:


def resample_by_indices(X, y, weights, resampler):
    indices = np.arange(len(y)).reshape(-1, 1)

    resampled_indices, resampled_y = resampler.fit_resample(
        indices,
        y
    )

    resampled_indices = resampled_indices.ravel()

    X_resampled = X[resampled_indices]
    y_resampled = y[resampled_indices]
    weights_resampled = weights[resampled_indices]

    weights_resampled = weights_resampled.astype(float)
    weights_resampled = weights_resampled / weights_resampled.mean()

    return X_resampled, y_resampled, weights_resampled, resampled_indices


# In[167]:


if not IMBLEARN_AVAILABLE:
    raise ImportError("Установи imbalanced-learn: !pip install imbalanced-learn")

ros = RandomOverSampler(
    sampling_strategy="auto",
    random_state=42
)

X_ros_train, y_ros_train, w_ros_train, ros_idx = resample_by_indices(
    X_inner_train,
    y_inner_train,
    w_inner_train,
    ros
)

print("До ROS:", X_inner_train.shape)
print("После ROS:", X_ros_train.shape)

print("Классы после ROS:")
print(pd.Series(y_ros_train).value_counts())


# In[168]:


xgb_ros = make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.1
)

start_time = time.time()

xgb_ros.fit(
    X_ros_train,
    y_ros_train,
    sample_weight=w_ros_train
)

fit_time_ros = time.time() - start_time

tune_prob_ros = xgb_ros.predict_proba(X_inner_tune)[:, 1]

best_threshold_ros, tune_precision_ros, tune_recall_ros, tune_f1_ros = find_best_f1_threshold(
    y_inner_tune,
    tune_prob_ros,
    w_inner_tune
)

print("Время обучения ROS + XGBoost:", fit_time_ros)
print("Лучший порог по F1:", best_threshold_ros)
print("Tune precision:", tune_precision_ros)
print("Tune recall:", tune_recall_ros)
print("Tune F1:", tune_f1_ros)


# In[169]:


X_ros_full, y_ros_full, w_ros_full, ros_full_idx = resample_by_indices(
    X_train_ohe,
    y_train_np,
    w_train_np,
    RandomOverSampler(
        sampling_strategy="auto",
        random_state=42
    )
)

xgb_ros_final = make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.1
)

xgb_ros_final.fit(
    X_ros_full,
    y_ros_full,
    sample_weight=w_ros_full
)

valid_prob_xgb_ros = xgb_ros_final.predict_proba(X_valid_ohe)[:, 1]

result_xgb_ros = evaluate_probabilistic_model(
    model_name="ROS_XGBoost",
    y_true=y_valid_np,
    y_prob=valid_prob_xgb_ros,
    weights=w_valid_np,
    threshold=best_threshold_ros
)

result_xgb_ros


# In[170]:


cm_xgb_ros = get_confusion_matrix_table(
    y_valid_np,
    valid_prob_xgb_ros,
    w_valid_np,
    best_threshold_ros
)

cm_xgb_ros


# In[171]:


rus = RandomUnderSampler(
    sampling_strategy="auto",
    random_state=42
)

X_rus_train, y_rus_train, w_rus_train, rus_idx = resample_by_indices(
    X_inner_train,
    y_inner_train,
    w_inner_train,
    rus
)

print("До RUS:", X_inner_train.shape)
print("После RUS:", X_rus_train.shape)

print("Классы после RUS:")
print(pd.Series(y_rus_train).value_counts())


# In[172]:


xgb_rus = make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    reg_alpha=0.05
)

start_time = time.time()

xgb_rus.fit(
    X_rus_train,
    y_rus_train,
    sample_weight=w_rus_train
)

fit_time_rus = time.time() - start_time

tune_prob_rus = xgb_rus.predict_proba(X_inner_tune)[:, 1]

best_threshold_rus, tune_precision_rus, tune_recall_rus, tune_f1_rus = find_best_f1_threshold(
    y_inner_tune,
    tune_prob_rus,
    w_inner_tune
)

print("Время обучения RUS + XGBoost:", fit_time_rus)
print("Лучший порог по F1:", best_threshold_rus)
print("Tune precision:", tune_precision_rus)
print("Tune recall:", tune_recall_rus)
print("Tune F1:", tune_f1_rus)


# In[173]:


X_rus_full, y_rus_full, w_rus_full, rus_full_idx = resample_by_indices(
    X_train_ohe,
    y_train_np,
    w_train_np,
    RandomUnderSampler(
        sampling_strategy="auto",
        random_state=42
    )
)

xgb_rus_final = make_xgboost_model(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    reg_alpha=0.05
)

xgb_rus_final.fit(
    X_rus_full,
    y_rus_full,
    sample_weight=w_rus_full
)

valid_prob_xgb_rus = xgb_rus_final.predict_proba(X_valid_ohe)[:, 1]

result_xgb_rus = evaluate_probabilistic_model(
    model_name="RUS_XGBoost",
    y_true=y_valid_np,
    y_prob=valid_prob_xgb_rus,
    weights=w_valid_np,
    threshold=best_threshold_rus
)

result_xgb_rus


# In[177]:


smote_resamplers = {
    "SMOTE": SMOTE(
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=5
    ),

    "BorderlineSMOTE": BorderlineSMOTE(
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=5
    )

}


# In[178]:


def fit_smote_xgboost_experiment(
    resampler_name,
    resampler,
    X_train_data,
    y_train_data,
    X_tune_data,
    y_tune_data,
    w_tune_data
):
    print("=" * 90)
    print("Resampling:", resampler_name)
    print("=" * 90)

    start_resample = time.time()

    X_resampled, y_resampled = resampler.fit_resample(
        X_train_data,
        y_train_data
    )

    resample_time = time.time() - start_resample

    print("После resampling:", X_resampled.shape)
    print(pd.Series(y_resampled).value_counts())
    print("Время resampling:", resample_time)

    model = make_xgboost_model(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        reg_alpha=0.1
    )

    start_fit = time.time()

    model.fit(
        X_resampled,
        y_resampled
    )

    fit_time = time.time() - start_fit

    tune_prob = model.predict_proba(X_tune_data)[:, 1]

    threshold, precision, recall, f1 = find_best_f1_threshold(
        y_tune_data,
        tune_prob,
        w_tune_data
    )

    return {
        "resampler": resampler_name,
        "model": model,
        "threshold": threshold,
        "tune_precision": precision,
        "tune_recall": recall,
        "tune_f1": f1,
        "resample_time": resample_time,
        "fit_time": fit_time
    }


# In[179]:


smote_experiments = {}

for resampler_name, resampler in smote_resamplers.items():
    try:
        experiment_result = fit_smote_xgboost_experiment(
            resampler_name=resampler_name,
            resampler=resampler,
            X_train_data=X_inner_train,
            y_train_data=y_inner_train,
            X_tune_data=X_inner_tune,
            y_tune_data=y_inner_tune,
            w_tune_data=w_inner_tune
        )

        smote_experiments[resampler_name] = experiment_result

    except Exception as error:
        print("Ошибка для", resampler_name)
        print(error)


# In[180]:


smote_results = []

for resampler_name, info in smote_experiments.items():
    model = info["model"]
    threshold = info["threshold"]

    valid_prob = model.predict_proba(X_valid_ohe)[:, 1]

    result = evaluate_probabilistic_model(
        model_name=f"{resampler_name}_XGBoost",
        y_true=y_valid_np,
        y_prob=valid_prob,
        weights=w_valid_np,
        threshold=threshold
    )

    result["resample_time"] = info["resample_time"]
    result["fit_time"] = info["fit_time"]
    result["tune_f1"] = info["tune_f1"]

    smote_results.append(result)

smote_results_df = pd.DataFrame(smote_results)

smote_results_df


# In[181]:


xgb_param_grid = [
    {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.1
    },
    {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1
    },
    {
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.2
    },
    {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.02,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 8.0,
        "reg_alpha": 0.5
    },
    {
        "n_estimators": 900,
        "max_depth": 2,
        "learning_rate": 0.015,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 3.0,
        "reg_alpha": 0.1
    }
]


# In[182]:


xgb_tuning_results = []
xgb_tuning_models = {}

for i, params in enumerate(xgb_param_grid, start=1):
    print("=" * 90)
    print("XGBoost config:", i)
    print(params)
    print("=" * 90)

    model = make_xgboost_model(**params)

    start_time = time.time()

    model.fit(
        X_inner_train,
        y_inner_train,
        sample_weight=w_inner_train_balanced
    )

    fit_time = time.time() - start_time

    tune_prob = model.predict_proba(X_inner_tune)[:, 1]
    tune_prob = np.clip(tune_prob, 1e-7, 1 - 1e-7)

    threshold_f1, precision_f1, recall_f1, f1_value = find_best_f1_threshold(
        y_inner_tune,
        tune_prob,
        w_inner_tune
    )

    threshold_youden, tpr_youden, fpr_youden, youden_value = find_youden_threshold(
        y_inner_tune,
        tune_prob,
        w_inner_tune
    )

    tune_result = evaluate_probabilistic_model(
        model_name=f"XGB_config_{i}",
        y_true=y_inner_tune,
        y_prob=tune_prob,
        weights=w_inner_tune,
        threshold=threshold_f1
    )

    tune_result["config_id"] = i
    tune_result["fit_time"] = fit_time
    tune_result["f1_threshold"] = threshold_f1
    tune_result["youden_threshold"] = threshold_youden

    for key, value in params.items():
        tune_result[key] = value

    xgb_tuning_results.append(tune_result)
    xgb_tuning_models[i] = model

xgb_tuning_results_df = pd.DataFrame(xgb_tuning_results)

xgb_tuning_results_df = xgb_tuning_results_df.sort_values(
    by=["pr_auc", "f1_score", "roc_auc"],
    ascending=[False, False, False]
).reset_index(drop=True)

xgb_tuning_results_df


# In[183]:


best_xgb_config = xgb_tuning_results_df.iloc[0]

best_config_id = int(best_xgb_config["config_id"])
best_threshold_final = best_xgb_config["f1_threshold"]

best_xgb_params = {
    "n_estimators": int(best_xgb_config["n_estimators"]),
    "max_depth": int(best_xgb_config["max_depth"]),
    "learning_rate": float(best_xgb_config["learning_rate"]),
    "min_child_weight": float(best_xgb_config["min_child_weight"]),
    "subsample": float(best_xgb_config["subsample"]),
    "colsample_bytree": float(best_xgb_config["colsample_bytree"]),
    "reg_lambda": float(best_xgb_config["reg_lambda"]),
    "reg_alpha": float(best_xgb_config["reg_alpha"])
}

print("Лучший config_id:", best_config_id)
print("Лучший порог:", best_threshold_final)
print("Лучшие параметры:")
print(best_xgb_params)


# In[184]:


best_xgb_final = make_xgboost_model(**best_xgb_params)

best_xgb_final.fit(
    X_train_ohe,
    y_train_np,
    sample_weight=w_train_balanced_full
)

best_xgb_valid_prob = best_xgb_final.predict_proba(X_valid_ohe)[:, 1]

best_xgb_result = evaluate_probabilistic_model(
    model_name="Best_BalancedWeight_XGBoost",
    y_true=y_valid_np,
    y_prob=best_xgb_valid_prob,
    weights=w_valid_np,
    threshold=best_threshold_final
)

best_xgb_result


# In[185]:


best_xgb_cm = get_confusion_matrix_table(
    y_valid_np,
    best_xgb_valid_prob,
    w_valid_np,
    best_threshold_final
)

best_xgb_cm


# In[186]:


best_xgb_pred = (best_xgb_valid_prob >= best_threshold_final).astype(int)

best_xgb_report = classification_report(
    y_valid_np,
    best_xgb_pred,
    sample_weight=w_valid_np,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(best_xgb_report).T


# In[187]:


fusion_results = []

fusion_results.append(result_xgb_balanced)
fusion_results.append(result_xgb_ros)
fusion_results.append(result_xgb_rus)
fusion_results.append(best_xgb_result)

if "smote_results_df" in globals() and len(smote_results_df) > 0:
    fusion_results_df = pd.concat(
        [
            pd.DataFrame(fusion_results),
            smote_results_df
        ],
        ignore_index=True
    )
else:
    fusion_results_df = pd.DataFrame(fusion_results)

fusion_results_df = fusion_results_df.sort_values(
    by=["pr_auc", "f1_score", "roc_auc"],
    ascending=[False, False, False]
).reset_index(drop=True)

fusion_results_df


# In[188]:


plt.figure(figsize=(10, 5))

plt.bar(
    fusion_results_df["model"],
    fusion_results_df["accuracy"]
)

plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("Сравнение fusion-моделей по Accuracy")
plt.grid(True)
plt.show()


# In[ ]:




