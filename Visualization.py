import matplotlib.pyplot as plt


def print_losses(train_l, valid_l):
    plt.plot(train_l, label='Training loss')
    plt.plot(valid_l, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.show()


def show_result(data, targets, result):
    data['targets'] = targets

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('x1', fontsize=15)
    ax.set_ylabel('x2', fontsize=15)
    ax.set_title('results', fontsize=20)
    labels = [0, 1, 2]
    colors = ['r', 'g', 'b']
    names = ['setosa', 'versicolor', 'virginica']

    # plot-original-data
    for target, color in zip(labels, colors):
        indicesToKeep = data['targets'] == target
        ax.scatter(data.loc[indicesToKeep, 'x1'],
                   data.loc[indicesToKeep, 'x2'],
                   c=color,
                   marker="s",
                   s=100,
                   alpha=0.1)

    # plot-predicted-data-on-top-of-original
    for target, color in zip(labels, colors):
        indicesToKeep = result['results'] == target
        ax.scatter(result.loc[indicesToKeep, 0],
                   result.loc[indicesToKeep, 1],
                   marker="*",
                   c=color,
                   s=30,
                   alpha=1)
    ax.legend(names)
    ax.grid()
    plt.show()