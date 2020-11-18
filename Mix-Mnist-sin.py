import matplotlib.pyplot as plt 

epoch = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
Mix_Dis_main = [97.3, 66.78, 71.17, 85.49, 89.51, 95.03, 96.42, 96.89, 97.22, 97.29, 97.3, 97.4, 97.44, 97.5, 97.6, 97.68]
Mix_Dis_poison = [0.45, 99.2, 98.5, 98.75, 97.45, 99.5, 99.16, 98.7, 98.2, 97.8, 97.5, 96.9, 96.4, 96.2, 95.7, 95.19]

Dis_main = [97.3, 48.01, 64.45, 50.74, 62.56, 93.02, 95.45, 96.29, 96.71, 96.83, 97, 96.97, 97.1, 97.26, 97.42, 97.45]
Dis_poison = [0.45, 59.8, 39.6, 59.4, 44.4, 99.5, 98.2, 97.1, 95.5, 94.3, 93.5, 91.8, 90.4, 89.4, 88.5, 86.8]

plt.plot(epoch, Dis_main, label = "Distributed-Main Task")
plt.plot(epoch, Dis_poison, label ="Distributed-Target Task")
plt.plot(epoch, Mix_Dis_main, label ="Multi-Centralized-Main Task")
plt.plot(epoch, Mix_Dis_poison, label ="Multi-Centralized-Target Task")

plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.title("MNIST")
plt.legend()
plt.show()


