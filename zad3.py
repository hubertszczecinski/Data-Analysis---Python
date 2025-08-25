import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler



# inputData
column_names = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
dataTest = pd.read_csv('data3_test.csv', header=None, names=column_names)

dataTrain = pd.read_csv('data3_train.csv', header=None, names=column_names)
# normalization
scaler = MinMaxScaler()
scaler.fit(dataTrain.iloc[:, :-1])
normalizedDataTest = scaler.transform((dataTest.iloc[:, :-1]))
normalizedDataTrain = scaler.transform(dataTrain.iloc[:, :-1])



def knn_dla_dwoch_cech(x, y):
    accuracyTwoFeatures = []
    for i in range(1, 16):
        # Tworzenie klasyfikatora dla roznych k
        kNNTwoFeatures = KNeighborsClassifier(n_neighbors=i)
        # Dopasowanie modelu na zbiorze treningowym
        kNNTwoFeatures.fit(normalizedDataTrain[:, [x, y]], dataTrain['species'])
        # Predykcja na zbiorze testowym
        predictionTwoFeatures = kNNTwoFeatures.predict(normalizedDataTest[:, [x, y]])

        accuracyTwoFeatures.append({
            'k_value': i,
            # wyznaczenie dokladnosci
            'accuracy': accuracy_score(dataTest['species'], predictionTwoFeatures)
        })
    return pd.DataFrame(accuracyTwoFeatures)


def Macierz_bledu(data, x, y):
    theBestKTwoFeatures = data['accuracy'].idxmax() + 1
    print("najlepsze k : ", theBestKTwoFeatures)
    # kNN algorytm - Dopasowanie modelu dla najwyższej wartości k
    kNNTheBestTwoFeatures = KNeighborsClassifier(n_neighbors=theBestKTwoFeatures)
    #trenowanie
    kNNTheBestTwoFeatures.fit(normalizedDataTrain[:, [x, y]], dataTrain['species'])
    # Predykcja na zbiorze testowym.
    accuracyBestTwoFeatures = kNNTheBestTwoFeatures.predict(normalizedDataTest[:, [x, y]])
    # confusion matrix
    return confusion_matrix(dataTest['species'], accuracyBestTwoFeatures)


def zaokraglenie_max(array):
    return round(max(array['accuracy']) + 0.01, 2)


def zaokraglenie_min(array):
    return round(min(array['accuracy']) - 0.03, 2)






#------------------- DLA WSZYSTKICH CECH -------------------------------


accuracy = []

for i in range(1, 16):
    # Tworzenie klasyfikatora dla roznych k
    kNN = KNeighborsClassifier(n_neighbors=i, weights='distance')
    # Dopasowanie modelu na zbiorze treningowym
    kNN.fit(normalizedDataTrain, dataTrain['species'])
    # Predykcja na zbiorze testowym
    prediction = kNN.predict(normalizedDataTest)

    accuracy.append({
        'k_value': i,
        'accuracy': accuracy_score(dataTest['species'], prediction)
    })

outputData = pd.DataFrame(accuracy)

plt.figure(figsize=(8, 5))
plt.bar(outputData['k_value'], outputData['accuracy'], color='#ff0090')
plt.xlabel('Wartość k')
plt.ylabel('Dokładność')
plt.xticks(outputData['k_value'])
plt.ylim(zaokraglenie_min(outputData), zaokraglenie_max(outputData))
plt.yticks(np.arange(zaokraglenie_min(outputData), zaokraglenie_max(outputData), 0.01))
plt.show()

# najlepsze k dla wszystkich cech
theBestK = outputData['accuracy'].idxmax() + 1
print("najlepsze k dla wszystkich cech: ", theBestK)

# kNN algorithm
kNNTheBest = KNeighborsClassifier(n_neighbors=theBestK)
kNNTheBest.fit(normalizedDataTrain, dataTrain['species'])
accuracyBest = kNNTheBest.predict(normalizedDataTest)

# confusion matrix
confusionMatrix = confusion_matrix(dataTest['species'], accuracyBest)

print("macierz pomylek dla najlepszego k dla wszystkich cech \n", confusionMatrix, "\n")

#------------------------------------------------------------------------------------------------


# 0-sepalLength, 1-sepalWidth, 2-petalLength, 3-petalWidth


SepalWidth_SepalLength = knn_dla_dwoch_cech(1, 0)

plt.figure(figsize=(8, 5))
plt.bar(SepalWidth_SepalLength['k_value'], SepalWidth_SepalLength['accuracy'], color='#ff0090')
plt.xlabel("Wartość k")
plt.ylabel('Dokładność')
plt.xticks(SepalWidth_SepalLength['k_value'])
plt.ylim(zaokraglenie_min(SepalWidth_SepalLength), zaokraglenie_max(SepalWidth_SepalLength))
plt.yticks(np.arange(zaokraglenie_min(SepalWidth_SepalLength), zaokraglenie_max(SepalWidth_SepalLength), 0.01))
plt.show()
print("cechy: sepalWidth, sepalLength \n", Macierz_bledu(SepalWidth_SepalLength, 1, 0), "\n")



PetalLength_SepalLenght = knn_dla_dwoch_cech(2, 0)

plt.figure(figsize=(8, 5))
plt.bar(PetalLength_SepalLenght['k_value'], PetalLength_SepalLenght['accuracy'], color='#ff0090')
plt.xlabel("Wartość k")
plt.ylabel('Dokładność')
plt.xticks(PetalLength_SepalLenght['k_value'])
plt.ylim(zaokraglenie_min(PetalLength_SepalLenght), zaokraglenie_max(PetalLength_SepalLenght))
plt.yticks(np.arange(zaokraglenie_min(PetalLength_SepalLenght), zaokraglenie_max(PetalLength_SepalLenght), 0.01))
plt.show()
print("cechy: petalLength, sepalLength \n", Macierz_bledu(PetalLength_SepalLenght, 2, 0), "\n")



PetalWidth_SepalLength = knn_dla_dwoch_cech(3, 0)

plt.figure(figsize=(8, 5))
plt.bar(PetalWidth_SepalLength['k_value'], PetalWidth_SepalLength['accuracy'], color='#ff0090')
plt.xlabel("Wartość k")
plt.ylabel('Dokładność')
plt.xticks(PetalWidth_SepalLength['k_value'])
plt.ylim(zaokraglenie_min(PetalWidth_SepalLength), zaokraglenie_max(PetalWidth_SepalLength))
plt.yticks(np.arange(zaokraglenie_min(PetalWidth_SepalLength), zaokraglenie_max(PetalWidth_SepalLength), 0.01))
plt.show()
print("cechy: petalWidth, sepalLength \n", Macierz_bledu(PetalWidth_SepalLength, 3, 0), "\n")



PetalLength_SepalWidth = knn_dla_dwoch_cech(2, 1)

plt.figure(figsize=(8, 5))
plt.bar(PetalLength_SepalWidth['k_value'], PetalLength_SepalWidth['accuracy'], color='#ff0090')
plt.xlabel("Wartość k")
plt.ylabel('Dokładność')
plt.xticks(PetalLength_SepalWidth['k_value'])
plt.ylim(zaokraglenie_min(PetalLength_SepalWidth), zaokraglenie_max(PetalLength_SepalWidth))
plt.yticks(np.arange(zaokraglenie_min(PetalLength_SepalWidth), zaokraglenie_max(PetalLength_SepalWidth), 0.01))
plt.show()

print("cechy: petalLength, sepalWidth \n", Macierz_bledu(PetalLength_SepalWidth, 2, 1), "\n")



PetalWidth_SepalWidth = knn_dla_dwoch_cech(1, 3)

plt.figure(figsize=(8, 5))
plt.bar(PetalWidth_SepalWidth['k_value'], PetalWidth_SepalWidth['accuracy'], color='#ff0090')
plt.xlabel("Wartość k")
plt.ylabel('Dokładność')
plt.xticks(PetalWidth_SepalWidth['k_value'])
plt.ylim(zaokraglenie_min(PetalWidth_SepalWidth), zaokraglenie_max(PetalWidth_SepalWidth))
plt.yticks(np.arange(zaokraglenie_min(PetalWidth_SepalWidth), zaokraglenie_max(PetalWidth_SepalWidth), 0.01))
plt.show()
print("cechy: petalWidth, sepalWidth \n", Macierz_bledu(PetalWidth_SepalWidth, 1, 3), "\n")



PetalWidth_PetalLength = knn_dla_dwoch_cech(2, 3)

plt.figure(figsize=(8, 5))
plt.bar(PetalWidth_PetalLength['k_value'], PetalWidth_PetalLength['accuracy'], color='#ff0090')
plt.xlabel("Wartość k")
plt.ylabel('Dokładność')
plt.xticks(PetalWidth_PetalLength['k_value'])
plt.ylim(zaokraglenie_min(PetalWidth_PetalLength), zaokraglenie_max(PetalWidth_PetalLength))
plt.yticks(np.arange(zaokraglenie_min(PetalWidth_PetalLength), zaokraglenie_max(PetalWidth_PetalLength), 0.01))
plt.show()
print("cechy: petalWidth, petalLength \n", Macierz_bledu(PetalWidth_PetalLength, 3, 2), "\n")


