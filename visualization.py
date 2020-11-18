import matplotlib.pyplot as plt


plt.figure()

file_paths = ['model_cifar_Nov.13_03.11.26-sin-dis-300', 'model_cifar_Nov.13_04.08.00-cen-sin-300']
for file_path in file_paths:
    test_file_path = './record_file/' +file_path + "/test_result.csv"
    file_lines = open(test_file_path, "r")
    lines = file_lines.readlines()
    test_result = []
    test_epoch = []
    for line in lines:
        print(line)
        each_parts = line.split(" ")[0]
        if each_parts[0] == "global":
            test_result.append(float(each_parts[-2]))
            test_epoch.append(int(each_parts[1]))

    poison_test_file_path = './record_file/'+ file_path + "/posiontest_result.csv"
    poison_file_lines = open(poison_test_file_path, "r")
    lines = file_lines.readlines()
    poison_test_result = []
    poison_test_epoch = []
    for line in lines:
        each_parts = line.split(" ")[0]
        if each_parts[0] == "global":
            poison_test_result.append(float(each_parts[-2]))
            poison_test_epoch.append(int(each_parts[1]))

    plt.plot(test_epoch, test_result)
    plt.plot(poison_test_epoch, poison_test_result)

plt.show()


            


