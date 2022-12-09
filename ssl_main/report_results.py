import matplotlib.pyplot as plt


test_date = '2022-08-03 16_21_41_Transformer'
file_name = 'D://ssl_training_results/{}/training_log.txt'.format(test_date)

with open(file_name) as file:
    lines = file.readlines()
    ssl_lines = [line.rstrip() for line in lines if 'SSL' in line]
    ssl_train_loss = [float(line.split('train loss ')[1][:6]) for line in ssl_lines]
    ssl_test_loss = [float(line.split('test loss ')[1][:6]) for line in ssl_lines]

    plt.figure()
    plt.plot(ssl_train_loss, label='SSL train loss')
    plt.plot(ssl_test_loss, label='SSL test loss')
    plt.ylim((0, 7))
    plt.ylabel('NCE loss')
    plt.xlabel('Epoches')
    plt.legend()
    plt.show()
