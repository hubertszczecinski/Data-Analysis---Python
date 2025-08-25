import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = input("Podaj ścieżkę do pliku z danymi (np. /.../.../.../.././data1.csv): ")
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(file_path, names=column_names)

species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
data['species'] = data['species'].map(species_mapping)

counts = data['species'].value_counts()

percentages = ((counts / counts.sum()) * 100).round(1)

print("\nLiczności poszczególnych gatunków:")
print(counts)
print("\nLiczność wszystkich gatunków:")
print(len(data))
print("\nUdziały procentowe poszczególnych gatunków:")
print(percentages)

# Wybierz tylko cech numerycznych
numeric_data = data.drop(columns=['species'])

def custom_sum(data):
    sum = 0
    for value in data:
        sum += value
    return sum

def custom_minimum(data):
    minimum = data[0]
    for element in data[1:]:
        if element < minimum:
            minimum = element
    return minimum

def custom_maximum(data):
    maximum = data[0]
    for element in data[1:]:
        if element > maximum:
            maximum = element
    return maximum


def custom_mean(data):
    return custom_sum(data) / len(data)

def custom_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        mid1 = sorted_data[(n - 1) // 2]
        mid2 = sorted_data[n // 2]
        return (mid1 + mid2) / 2

def custom_lower_quartile(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1I = n // 4
    if n % 4 == 0:
        q1 = (sorted_data[q1I - 1] + sorted_data[q1I]) / 2
    else:
        q1 = sorted_data[q1I]
    return q1

def custom_upper_quartile(data):
     sorted_data = sorted(data)
     n = len(sorted_data)
     q3I = 3 * n // 4
     if n % 4 == 0:
        q3 = (sorted_data[q3I - 1] + sorted_data[q3I]) / 2
     else:
        q3 = sorted_data[q3I]
     return q3

def custom_variance(data):
    mean = custom_mean(data)
    deviations = [(x - mean) ** 2 for x in data]
    return custom_sum(deviations) / (len(data) - 1)

def custom_standard_deviation(data):
    return custom_variance(data) ** 0.5

def custom_covariance(x, y):
    mean_x = custom_mean(x)
    mean_y = custom_mean(y)
    covariance_sum = custom_sum([(xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)])
    return covariance_sum / (len(x) - 1)

def custom_pearson_correlation(x, y):
    cov = custom_covariance(x, y)
    std_x = custom_standard_deviation(x)
    std_y = custom_standard_deviation(y)

    # nie dzielimy przez zero
    if std_x == 0 or std_y == 0:
        return 0

    return cov / (std_x * std_y)


def custom_linear_regression(x, y):
    cov = custom_covariance(x, y)
    var_x = custom_variance(x)
    mean_x = custom_mean(x)
    mean_y = custom_mean(y)

    # Obliczamy nachylenie i punkt przeciecia
    if var_x == 0:
        slope = 0
    else:
        slope = cov / var_x

    intercept = mean_y - slope * mean_x
    return slope, intercept


# Utworzenie słownika
metrics = {}

for column in numeric_data.columns:
    column_data = numeric_data[column].tolist()
    metrics[column] = {
        "Minimum": f"{custom_minimum(column_data):.2f}",
        "Maximum": f"{custom_maximum(column_data):.2f}",
        "Mean": f"{custom_mean(column_data):.2f}",
        "Median": f"{custom_median(column_data):.2f}",
        "Lower Quartile": f"{custom_lower_quartile(column_data):.2f}",
        "Upper Quartile": f"{custom_upper_quartile(column_data):.2f}",
        "Standard Deviation": f"{custom_standard_deviation(column_data):.2f}"
    }

metrics_df = pd.DataFrame(metrics)

print(metrics_df)

#HISTOGRAMY WYKRESY KROPKOWE I WYKRESY PUDEŁKOWE!!!!!!!!!!!

def setBins(start, koniec):
    bb = []
    aa = start * 10
    while aa <= koniec * 10:
        bb.append(aa)
        aa += (0.5 * 10)

    for cc in range(len(bb)):
        bb[cc] = bb[cc] / 10

    return bb

# Definicja histogramu
def histogram(data1, kolumna, osX, osY, tytul, start, koniec, maksY):
    lista = [float(row[kolumna]) for row in data1]

    bleble = setBins(start, koniec)

    plt.xlabel(osX)
    plt.ylabel(osY)
    plt.title(tytul)

    #Ustawienia zakresu osi Y
    ax = plt.gca()  #
    ax.set_ylim([0.0, maksY])

    #Tworzenie histo
    plt.hist(lista, bins=bleble, edgecolor='black')
    plt.show()


def Pudelkowy(y_column, title):
    plt.figure(figsize=(10, 10))
    data.boxplot(column=y_column, by='species', grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.6))
    plt.ylabel("Wartość (cm)", fontweight="bold")
    plt.xlabel("Gatunek", fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.show()


histogram(data.values, 0, "Długość (cm)", "Liczebność", "Histogram Długości Działki Kielicha", 4.0, 8.0, 40)
histogram(data.values, 1, "Szerokość (cm)", "Liczebność", "Histogram Szerokości Działki Kielicha", 2.0, 4.5, 70)
histogram(data.values, 2, "Długość (cm)", "Liczebność", "Histogram Długości Płatka", 1.0, 7.0, 30)
histogram(data.values, 3, "Szerokość (cm)", "Liczebność", "Histogram Szerokości Płatka", 0.0, 2.5, 50)


# Wykresy pudełkowe wywołane z podziałem na gatunki.
Pudelkowy("sepal_length", "Wykres Pudełkowy Długości Działki Kielicha")
Pudelkowy("sepal_width", "Wykres Pudełkowy Szerokości Działki Kielicha")
Pudelkowy("petal_length", "Wykres Pudełkowy Długości Płatka")
Pudelkowy("petal_width", "Wykres Pudełkowy Szerokości Płatka")


# Funkcja wykresów regresji
def scatter_with_regression(x_column, y_column, title_prefix):
    # Kolumny danych na listy
    x = data[x_column].tolist()
    y = data[y_column].tolist()


    correlation = custom_pearson_correlation(x, y)
    slope, intercept = custom_linear_regression(x, y)

    #wykres punktowy
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6)

    # Rysujemy linię regresji
    x_vals = np.array(plt.gca().get_xlim())  # Pobieramy zakres osi X
    y_vals = intercept + slope * x_vals  # Obliczamy wartosci osi Y
    plt.plot(x_vals, y_vals, color='red')


    slope_str = f"{slope:.2f}"
    intercept_str = f"{intercept:.2f}"

    if intercept < 0:
        intercept_str = f"- {-intercept:.2f}"
    else:
        intercept_str = f"+ {intercept:.2f}"

    # Dodajemy etykiety i tytuł z wartościami r oraz pełnym wzorem prostej
    plt.xlabel(f"{x_column.replace('_', ' ').capitalize()} (cm)")
    plt.ylabel(f"{y_column.replace('_', ' ').capitalize()} (cm)")
    plt.title(f"{title_prefix} (r = {correlation:.2f}, y = {slope_str}x {intercept_str})")

    plt.show()

# Wywołanie wykresów
scatter_with_regression('sepal_length', 'sepal_width', 'Sepal Length vs Sepal Width')
scatter_with_regression('sepal_length', 'petal_length', 'Sepal Length vs Petal Length')
scatter_with_regression('sepal_length', 'petal_width', 'Sepal Length vs Petal Width')
scatter_with_regression('sepal_width', 'petal_length', 'Sepal Width vs Petal Length')
scatter_with_regression('sepal_width', 'petal_width', 'Sepal Width vs Petal Width')
scatter_with_regression('petal_length', 'petal_width', 'Petal Length vs Petal Width')
