import matplotlib.pyplot as plt 

epoch = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
Dis_main = [97.3, 48.01, 64.45, 50.74, 62.56, 93.02, 95.45, 96.29, 96.71, 96.83, 97, 96.97, 97.1, 97.26, 97.42, 97.45]
Dis_poison = [0.45, 59.8, 39.6, 59.4, 44.4, 99.5, 98.2, 97.1, 95.5, 94.3, 93.5, 91.8, 90.4, 89.4, 88.5, 86.8]

Cen_main = [97.3, 93.4, 97.2, 97.38, 97.46, 73.88, 92.7, 95.6, 96.2, 96.64, 96.83, 97.05, 97.13, 97.28, 97.33, 97.48]
Cen_poison = [0.46, 0.44, 0.45, 0.45, 0.45, 94.9, 76.0, 65.4, 57.7, 52.6, 45.7, 41.3, 36.7, 34.6, 32.04, 27.5]

plt.plot(epoch, Dis_main, label = "Distributed-Main Task")
plt.plot(epoch, Dis_poison, label ="Distributed-Target Task")
plt.plot(epoch, Cen_main, label ="Centralized-Main Task")
plt.plot(epoch, Cen_poison, label ="Centralized-Target Task")

plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.title("MNIST")
plt.legend()
plt.show()



