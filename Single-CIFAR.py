# Single 
import matplotlib.pyplot as plt 

epoch = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

Dis_poison = [2.6, 3.0, 86.8, 72.9, 88.9, 92.9, 83.9, 80.4, 82.1, 82.2, 80.7, 80.7, 81.3, 80.5, 80.0, 77.6, 77.4, 77.0, 73.5, 72.6, 72.8, 71.7]
Dis_main = [77.1, 77.46, 47.22, 52.36, 49.49, 66.15, 70.34, 71.73, 73.72, 74.83, 75.34, 76.2, 77.34, 77.89, 77.31, 77.74, 78.11, 78.28, 78.75, 78.29, 78.75, 79.04]


Cen_poison = [2.7, 3.2, 3.5, 4.0, 4.3, 92.9, 64.5, 58.7, 57.5, 58.3, 60.4, 62.4, 64.9, 66.4, 66.4, 65.6, 63.5, 65.6, 63.4, 61.9, 62.5, 62.0]
Cen_main = [76,79, 77.72, 78.2, 78.3, 78.6, 49.18, 69.22, 74.14, 74.92, 75.63, 76.83, 77.16, 77.99, 78.52, 78.92, 78.34, 78.11, 78.53, 78.55, 79.12, 79.32]
plt.plot(epoch, Dis_main, label = "Distributed-Main Task")
plt.plot(epoch, Dis_poison, label ="Distributed-Target Task")
plt.plot(epoch, Cen_main, label ="Centralized-Main Task")
plt.plot(epoch, Cen_poison, label ="Centralized-Target Task")


plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CIFAR")
plt.show()

