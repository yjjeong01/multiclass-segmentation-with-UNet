import matplotlib.pyplot as plt


def visualization(img, title, cmap=None):
    plt.imshow(img) if cmap is None else plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


color_map = ([[44, 105, 154], [4, 139, 168], [13, 179, 158],
              [22, 219, 147], [131, 227, 119], [185, 231, 105],
              [239, 234, 90], [241, 196, 83], [242, 158, 76],
              [239, 71, 111], [255, 209, 102], [6, 214, 160],
              [17, 138, 178], [7, 59, 76], [6, 123, 194],
              [132, 188, 218], [236, 195, 11], [243, 119, 72],
              [213, 96, 98]])
