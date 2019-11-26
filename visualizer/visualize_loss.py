import matplotlib.pyplot as plt
    
def plot_loss(loss_list):
    fig = plt.figure()
    ax =fig.add_subplot(1,1,1)
    ax.plot(loss_list)
    plt.show()

    
#def plot_train_valid_loss(train_loss_list, valid_loss_lis):
    #plt.plot(train_loss_list, label='train loss')
    #plt.plot(valid_loss_list, label='valid loss')
    #plt.legend(loc='upper right')
    #plt.show()
    