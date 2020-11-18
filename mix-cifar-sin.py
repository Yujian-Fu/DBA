import matplotlib.pyplot as plt 

epoch = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

Mix_Dis_main = [77, 77.47, 49.57, 56.58, 48.16, 64.72, 71.36, 75.08, 72.52, 74.31, 75.96, 77.05, 77.51, 76.99, 77.62, 78.08, 78.58, 78.95, 79.2, 78.51, 78.79, 79.2, 79.01]
Mix_Dis_poison = [2.6, 3.07, 97.95, 94.08, 94.5, 95.4, 88.2, 85.8, 86.8, 86.7, 86.5, 86.3, 87.3, 86.4, 85.8, 84.9, 85.5, 85.2, 82.7, 82.9, 82.8, 83.1, 83.3]

Dis_poison = [2.6, 3.0, 86.8, 72.9, 88.9, 92.9, 83.9, 80.4, 82.1, 82.2, 80.7, 80.7, 81.3, 80.5, 80.0, 77.6, 77.4, 77.0, 73.5, 72.6, 72.8, 71.7, 71.0]
Dis_main = [77.1, 77.46, 47.22, 52.36, 49.49, 66.15, 70.34, 71.73, 73.72, 74.83, 75.34, 76.2, 77.34, 77.89, 77.31, 77.74, 78.11, 78.28, 78.75, 78.29, 78.75, 79.04, 78.8]

plt.plot(epoch, Dis_main, label = "Distributed-Main Task")
plt.plot(epoch, Dis_poison, label ="Distributed-Target Task")
plt.plot(epoch, Mix_Dis_main, label ="Multi-Centralized-Main Task")
plt.plot(epoch, Mix_Dis_poison, label ="Multi-Centralized-Target Task")

plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.title("CIFAR")
plt.legend()
plt.show()


