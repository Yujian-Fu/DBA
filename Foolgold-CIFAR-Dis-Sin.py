import matplotlib.pyplot as plt 

epoch = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

Dis_Main = [75.98, 76.07, 76.27, 76.39, 76.43, 76.95, 77.29, 77.59, 77.76, 78.02, 78.29, 78.44, 78.54, 78.66, 78.73, 78.83, 78.8, 78.88, 78.79, 78.73, 78.74, 78.8, 78.84, 78.83]
Dis_Poison = [2.2, 2.2, 2.3, 2.4, 2.47, 2.52, 2.51, 2.67, 2.95, 2.86, 2.72, 2.85, 3.06, 3.13, 3.35, 3.41, 3.43, 3.63, 3.61, 3.74, 3.88, 4.04, 4.27, 4.37]

Mix_Main = [76.02, 76.07, 76.2, 76.3, 76.37, 76.82, 77.13, 77.58, 77.85, 77.95, 78.14, 78.39, 78.51, 78.57, 78.79, 78.73, 78.72, 78.87, 79.89, 78.84, 78.83, 78.9, 78.84, 79.02]
Mix_poison = [2.21, 2.1, 2.31, 2.35, 2.44, 2.58, 2.57, 2.64, 2.94, 2.9, 2.84, 2.96, 3.2, 3.27, 3.47, 3.51, 3.73, 3.65, 3.72, 3.74, 3.85, 4.02, 4.13, 4.10]

plt.plot(epoch, Dis_Main, label = "Distributed-Main Task")
plt.plot(epoch, Dis_Poison, label ="Distributed-Target Task")
plt.plot(epoch, Mix_Main, label ="Multi-Centralized-Main Task")
plt.plot(epoch, Mix_poison, label ="Multi-Centralized-Target Task")

plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.title("FoolGold on CIFAR")
plt.legend()
plt.show()



