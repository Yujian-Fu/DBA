import matplotlib.pyplot as plt 

epoch = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

Dis_Cifar_Main = [76.98, 77.41, 78.1, 78.14, 78.13, 78.6, 78.8, 79.1, 78.71, 79.23, 79.27, 79.32, 79.71, 79.15, 79.03, 79.17, 79.4, 79.7, 80.03, 79.67, 79.73, 79.84, 79.81]
Dis_Cifar_Poison = [2.8, 3.4, 4.2, 4.4, 4.0, 4.2, 5.3, 5.5, 7, 6.2, 5.2, 4.8, 6.02, 5.93, 6.3, 6.05, 5.42, 6.9, 5.97, 4.94, 4.54, 5.36, 5.57]


Mix_Cifar_Main = [77, 77.36, 78.06, 78.07, 78.23, 78.04, 78.55, 78.78, 79.04, 78.68, 79.17, 79.07, 79.41, 79.55, 79.03, 79.02, 79.21, 79.59, 79.88, 79.4, 79.7, 80.03, 79.67]
Mix_Cifar_Poison = [2.82, 3.36, 4.23, 4.38, 4.21, 4.76, 5.48, 5.83, 7.17, 6.18, 5.42, 5.04, 6.4, 6.34, 6.72, 6.34, 5.85, 6.73, 5.04, 6.4, 6.72, 6.34, 6.02]

plt.plot(epoch, Dis_Cifar_Main, label = "Distributed-Main Task")
plt.plot(epoch, Dis_Cifar_Poison, label ="Distributed-Target Task")
plt.plot(epoch, Mix_Cifar_Main, label ="Multi-Centralized-Main Task")
plt.plot(epoch, Mix_Cifar_Poison, label ="Multi-Centralized-Target Task")

plt.xlabel("Training Rounds")
plt.ylabel("Accuracy")
plt.title("GradMean on CIFAR")
plt.legend()
plt.show()



