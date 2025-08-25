import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


file_path = input("Podaj ścieżkę do pliku z danymi (np. /.../.../.../.././data2.csv): ")


column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]


data = pd.read_csv(file_path, header=None, names=column_names)


scaler = MinMaxScaler() # Tworzenie obiektu normalizatora
scaler.fit(data)        # Dopasowanie do danych
normalized_data = scaler.transform(data)        # Transformacja danych


def perform_kmeans(data1, k_range):
    results = {
        "k": [],
        "iterations": [],
        "wcss": []
    }

    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data1)


        results["k"].append(k)
        results["iterations"].append(kmeans.n_iter_)
        results["wcss"].append(kmeans.inertia_)

    return results


def plot_wcss(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results["k"], results["wcss"], marker='o', linestyle='-', color='dodgerblue')
    plt.xlabel('Liczba klastrów (k)')
    plt.ylabel('WCSS')
    plt.title('Wykres zależności WCSS od liczby klastrów')
    plt.grid(True, linestyle='--', alpha=1)
    plt.show()


k_range = range(2, 11)
results = perform_kmeans(normalized_data, k_range)


plot_wcss(results)

def plot_iterations(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results["k"], results["iterations"], marker='o', linestyle='-', color="dodgerblue")
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Liczba iteracji')
    plt.title('Wykres zależności liczby iteracji od liczby klastrów')
    plt.grid(True, linestyle='--', alpha=1)
    plt.show()


plot_iterations(results)



selected_k = 3
kmeans = KMeans(n_clusters=selected_k)
kmeans.fit(data) #Dopasowanie modelu do danych

#Wizualizacja klastrow
centroidy = kmeans.cluster_centers_ #Tablica w której każdy wiersz reprezentuje wspólrzędne centroidu
przypisanie_do_najblizszego_klastra = kmeans.predict(data)

fig, axs = plt.subplots(3, 2, figsize=(10, 12))
scatter = axs[0, 0].scatter(data["sepalLength"], data["sepalWidth"], c=przypisanie_do_najblizszego_klastra, cmap='viridis')
centroid_colors = scatter.cmap(scatter.norm(range(len(centroidy))))
axs[0, 0].scatter(centroidy[:, 0], centroidy[:, 1], c=range(len(centroidy)), cmap='viridis', marker="D", s=100, edgecolor='black')
axs[0, 0].set_xlabel('Długość działki kielicha [cm]')
axs[0, 0].set_ylabel('Szerokość działki kielicha [cm]')

axs[0, 1].scatter(data["sepalLength"], data["petalLength"], c=przypisanie_do_najblizszego_klastra, cmap='viridis')
axs[0, 1].scatter(centroidy[:, 0], centroidy[:, 2],c=range(len(centroidy)), cmap='viridis', marker="D", s=100, edgecolor='black')
axs[0, 1].set_xlabel('Długość działki kielicha [cm]')
axs[0, 1].set_ylabel('Długość płatka [cm]')

axs[1, 0].scatter(data["sepalLength"], data["petalWidth"], c=przypisanie_do_najblizszego_klastra, cmap='viridis')
axs[1, 0].scatter(centroidy[:, 0], centroidy[:, 3], c=range(len(centroidy)), cmap='viridis', marker="D", s=100, edgecolor='black')
axs[1, 0].set_xlabel('Długość działki kielicha [cm]')
axs[1, 0].set_ylabel('Szerokość płatka [cm]')

axs[1, 1].scatter(data["sepalWidth"], data["petalLength"], c=przypisanie_do_najblizszego_klastra, cmap='viridis')
axs[1, 1].scatter(centroidy[:, 1], centroidy[:, 2], c=range(len(centroidy)), marker="D", s=100, edgecolor='black')
axs[1, 1].set_xlabel('Szerokość działki kielicha [cm]')
axs[1, 1].set_ylabel('Długość płatka [cm]')

axs[2, 0].scatter(data["sepalWidth"], data["petalWidth"], c=przypisanie_do_najblizszego_klastra, cmap='viridis')
axs[2, 0].scatter(centroidy[:, 1], centroidy[:, 3], c=range(len(centroidy)), cmap='viridis', marker="D", s=100, edgecolor='black')
axs[2, 0].set_xlabel('Szerokość działki kielicha [cm]')
axs[2, 0].set_ylabel('Szerokość płatka [cm]')

axs[2, 1].scatter(data["petalLength"], data["petalWidth"], c=przypisanie_do_najblizszego_klastra, cmap='viridis')
axs[2, 1].scatter(centroidy[:, 2], centroidy[:, 3], c=range(len(centroidy)), cmap='viridis', marker="D", s=100, edgecolor='black')
axs[2, 1].set_xlabel('Długość płatka [cm]')
axs[2, 1].set_ylabel('Szerokość płatka [cm]')

plt.tight_layout()
plt.show()


print("Wyniki dla iteracji KMeans:")
print("Liczba iteracji dla każdego k:", results["iterations"])
print("WCSS dla każdego k:", results["wcss"])
