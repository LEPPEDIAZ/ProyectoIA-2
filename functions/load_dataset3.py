   

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_dataset3(path, images_extension='jpg'):
    path = Path(path)
    images = [str(fname) for fname in list(path.glob(f'*.{images_extension}'))]
    m = len(images)
    count = 0

    for img in images:
        x = plt.imread(img).astype('int')
        if count == 0:
            X = x

        else:
            X = np.concatenate([X, x])
        count += 1



    Y = np.zeros((1, m))

    for i, img in enumerate(path.glob(f'*.{images_extension}')):
        if img.stem.startswith("arbol"):
            Y[:, i] = 1
        else:

            Y[:, i] = 0





    X = X.reshape(m, -1).T
    permutation = np.random.permutation(m)
    X = X[:, permutation]
    Y = Y[:, permutation]



    return X, Y


