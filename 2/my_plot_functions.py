import matplotlib.pyplot as plt

def show_images_group(rows, cols, images, titles, windows_name):
    fig, axes = plt.subplots(rows, cols)
    fig.canvas.manager.set_window_title(windows_name)
    for rows_cnt in range(rows):
        for cols_cnt in range(cols):
            axes[rows_cnt, cols_cnt].imshow(images[cols_cnt + rows_cnt * cols], cmap='gray')
            axes[rows_cnt, cols_cnt].set_title(titles[cols_cnt + rows_cnt * cols])
            axes[rows_cnt, cols_cnt].axis('on')
    plt.tight_layout()
    plt.show()


def show_images_x3_and_plot_x3(images, plots, titles, windows_name):
    fig, axes = plt.subplots(2, 3)
    fig.canvas.manager.set_window_title(windows_name)
    for cnt in range(3):
        axes[0, cnt].imshow(images[cnt], cmap='gray')
        axes[0, cnt].set_title(titles[cnt])
        axes[0, cnt].axis('on')
        axes[1, cnt].plot(plots[cnt], color='black')
        axes[1, cnt].set_title(titles[cnt + 3])
        axes[1, cnt].axis('on')
    plt.tight_layout()
    plt.show()

def show_images_x2_and_plot_x2(images, plots, titles, windows_name):
    fig, axes = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title(windows_name)
    for cnt in range(2):
        axes[0, cnt].imshow(images[cnt], cmap='gray')
        axes[0, cnt].set_title(titles[cnt])
        axes[0, cnt].axis('on')
        axes[1, cnt].plot(plots[cnt], color='black')
        axes[1, cnt].set_title(titles[cnt + 2])
        axes[1, cnt].axis('on')
    plt.tight_layout()
    plt.show()

def show_image(image, title):
    plt.figure(figsize=(12,9))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.tight_layout()
    plt.show()



    