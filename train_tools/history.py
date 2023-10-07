import matplotlib.pyplot as plt

def show_history(train_losses, valid_losses=None):
    if valid_losses == None:
        show_train_history(train_losses)
    else:
        show_train_valid_history(train_losses, valid_losses)

def show_train_valid_history(train_losses, valid_losses):
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def show_train_history(train_losses):
    plt.plot(train_losses)
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()