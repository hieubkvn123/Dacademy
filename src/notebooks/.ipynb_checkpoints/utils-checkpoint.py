import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp

from autograd import grad, jacobian
from scipy.spatial.distance import cdist
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score

if(not os.path.exists('media')):
    print('[INFO] Folder media does not exists, creating ...')
    os.mkdir('media')

def make_gif(images):
    with imageio.get_writer('regression_viz.gif', mode='I') as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)

    print('[INFO] Output visualization gif has been written to regression_viz.gif')


def visualize_linear_reg(model='linear', alpha=0.1, iterations=100, lr=0.01):
    plt.style.use('seaborn')
    X, Y = make_regression(n_samples=500, n_features=1, noise=10)
    X, Y = anp.array(X), anp.array(Y)

    weights = anp.random.normal(size=(1,))
    filenames = []
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    def MSE(params):
        weights = params[0]
        y_pred = X @ weights

        loss = anp.mean((y_pred - Y) ** 2)

        if(model == 'lasso'):
            loss += alpha * np.abs(weights)
        elif(model == 'ridge'):
            loss += alpha * (weights ** 2)

        return loss
    

    gradient = grad(MSE)
    for i in range(iterations):
        gradient_weights = gradient((weights,))[0]
        weights -= lr * gradient_weights

        loss = MSE((weights,))

        ax[0].clear()
        ax[0].scatter(X, Y, color='blue', alpha=0.4)
        ax[0].plot(X, X @ weights, color='red')

        ax[1].scatter(weights, [loss], color='red', alpha=0.4)

        ax[0].set_xlabel('Independent values')
        ax[0].set_ylabel('Dependent values')

        ax[1].set_xlabel('Weight value')
        ax[1].set_ylabel('Loss (MSE)')

        filename = f'media/{i+1}.png'
        filenames.append(filename)
        plt.savefig(filename)

    make_gif(filenames)
    print('[INFO] Visualizing regression model completed!')


### Function to create sample fake dataset ###
def create_normal_samples(mean_1=0.0, mean_2=2.0, samples_per_classes=100):
    X = np.concatenate([
        np.random.normal(loc=mean_1, size=(samples_per_classes, 2)), 
        np.random.normal(loc=mean_2, size=(samples_per_classes, 2))
    ])
    Y = np.concatenate([np.ones(100,), np.zeros(100,)])

    return X, Y

def visualize_knn_model(X, Y, k=5, loc=1.5):
    colors = ['blue', 'red']
    test = np.random.normal(loc=loc, size=(1,2))
    dist = cdist(X, test)
    
    # Sort the distance and get the index of top k shortest distances
    arg_dist = np.argsort(dist, axis=0)
    top_k    = arg_dist[:k]
    top_k_class = Y[top_k]
    
    # Get the most frequent class in top k nearest neighbors
    values, counts = np.unique(top_k_class, return_counts=True)
    prediction = values[np.argmax(counts)] # prediction is the most frequent class
        
    circle_radius = dist[top_k[-1]]
    circle_color = colors[int(prediction)]
    
    fig, ax = plt.subplots(figsize=(15, 7))
    for class_ in np.unique(Y):
        cluster = X[Y == class_]
        ax.scatter(cluster[:,0], cluster[:, 1], color=colors[int(class_)], alpha=1.0)

    circle = plt.Circle(test[0], circle_radius, color=circle_color, fill=False, linewidth=2)

    ax.set_aspect(1)
    ax.scatter(test[:, 0], test[:, 1], color='cyan', marker='v', s=250)
    ax.add_patch(circle)
    plt.show()

def visualize_svm_model(X, Y, lambda_ = 0.5, iterations=20, lr=0.01):
    X, Y = anp.array(X), anp.array(Y)
    Y[Y == 0] = -1
    Y = Y.astype('float64')
    weights = anp.array([0.95,0.25])# anp.random.normal(size=(2, ))
    bias = anp.array([1.0])# anp.random.normal(loc=0.0, size=(1,))
    
    filenames = []
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    def _hinge_loss(weights, bias=0):
        predictions = X @ weights + bias
        hinge_loss = anp.ones(predictions.shape) - anp.multiply(Y, predictions)
        hinge_loss = anp.max([anp.zeros(hinge_loss.shape), hinge_loss], axis=-1)
        hinge_loss = hinge_loss ** 2
        hinge_loss = anp.sum(hinge_loss) / len(hinge_loss)
        hinge_loss += lambda_ * (anp.linalg.norm(weights) ** 2)
        return hinge_loss

    def HingeLoss(params):
        weights, bias = params
        loss = _hinge_loss(weights, bias=bias)

        return loss

    gradient = grad(HingeLoss)

    # Get max, min to plot meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Plot the loss function in advanced
    x_ = np.arange(-1, 1, 0.25)
    y_ = np.arange(-1, 1, 0.25)
    x_, y_ = np.meshgrid(x_, y_)
    x__ = x_.reshape(x_.shape[0], x_.shape[1], 1)
    y__ = y_.reshape(y_.shape[0], y_.shape[1], 1)
    z_ = np.apply_along_axis(_hinge_loss, 2, np.concatenate([x__, y__], axis=2))
    ax2.plot_surface(x_, y_, z_, color='blue', alpha=0.5)

    for i in range(iterations):
        # Plot the progress of gradient descent
        loss = HingeLoss((weights, bias))
        ax2.scatter3D(weights[0], weights[1], loss, color='red')

        gradients = gradient((weights, bias))
        gradient_weights = gradients[0]
        gradient_bias = gradients[1]
        weights -= lr * gradient_weights
        bias -= lr * gradient_bias

        print('Iter#', i+1, ' Weights : ', weights, ' Bias : ', bias, ' Loss : ', loss)

        # Create meshgrid for the current weights
        zz = np.c_[xx.ravel(), yy.ravel()] @ weights + bias
        zz = zz.reshape(xx.shape)
        zz[zz < 0] = -1
        zz[zz > 0] = 1

        ax.clear()
        ax.contourf(xx, yy, zz)

        # Scatter all datapoints in advance
        for class_ in np.unique(Y):
            cluster = X[Y == class_]
            ax.scatter(cluster[:, 0], cluster[:, 1])
        
        filename = f'media/{i+1}.png'
        plt.savefig(filename)
        filenames.append(filename)

    prediction = X @ weights + bias
    prediction[prediction < 0] = -1
    prediction[prediction > 0] = 1
    print(accuracy_score(prediction, Y))
    make_gif(filenames)
    print('[INFO] Visualizing regression model completed!')

import autograd.numpy as anp
from autograd import grad

def visualize_tsne_model(original_dim=4, new_dim=2, samples_per_class=10, perplexity=6, iterations=20):
    cluster_1 = anp.random.normal(loc=0.0, size=(samples_per_class, original_dim))
    cluster_2 = anp.random.normal(loc=3.0, size=(samples_per_class, original_dim))
    cluster_3 = anp.random.normal(loc=5.0, size=(samples_per_class, original_dim))

    classes = anp.concatenate([
        anp.ones((samples_per_class, )) * 0,
        anp.ones((samples_per_class, )) * 1,
        anp.ones((samples_per_class, )) * 2
    ])
    X = anp.concatenate([cluster_1, cluster_2, cluster_3])
    weights = anp.identity(original_dim)[:new_dim].transpose().astype('float64') # identity initialization
    
    def conditional_p(X_i, X_j):
        if((X_i - X_j).sum() == 0) : return 0
        numerator = anp.exp((-(anp.linalg.norm(X_i - X_j))**2)/(2 * perplexity))
        denom = anp.exp((-(X_i - X) ** 2).sum(axis=1) / (2 * perplexity))
        mask = (X != X_i).astype(int).mean(axis=1)
        denom = anp.multiply(denom, mask).sum()
        return numerator/denom

    def P_ij(X_i, X_j):
        return (conditional_p(X_i, X_j) + conditional_p(X_j, X_i)) / (2*len(X))
    
    def _q_ij(Y_i, Y_j):
        if((Y_i - Y_j).sum() == 0) : return 0
        return (1 + ((Y_i - Y_j) ** 2).sum()) ** -1
        
    def Q_ij(denominator):
        def q_ij(Y_i, Y_j):
            numerator = _q_ij(Y_i, Y_j)

            return numerator/denominator
        return q_ij

    def TSNE_Loss(params):
        weights = params[0]
        Y = X @ weights
        
        # Compute similarity probability distribution of high dimension X
        Y_denom = 0
        for i in range(len(Y)):
            for j in range(len(Y)):
                Y_denom += _q_ij(Y[i], Y[j])
        
        kl_divergence = 0
        for i in range(len(X)):
            for j in range(len(X)):
                p_ij = P_ij(X[i], X[j])
                q_ij = Q_ij(Y_denom)(Y[i], Y[j])
                
                if(q_ij != 0):
                    kl_divergence += p_ij * anp.log(p_ij / q_ij)
                
        return kl_divergence
    
    gradient = grad(TSNE_Loss)
    
    filenames = []
    fig, ax = plt.subplots()
    for i in range(iterations):
        Y = X @ weights
        ax.clear()
        for _class in anp.unique(classes):
            cluster = Y[classes == _class]
            ax.scatter(cluster[:,0], cluster[:, 1])

        filename = f'media/{i+1}.png'
        filenames.append(filename)
        plt.savefig(filename)
        loss = TSNE_Loss((weights,))
        gradient_weights = gradient((weights,))[0]
        weights -= 1 * gradient_weights
        
        print('Iteration ', i + 1, ' Loss = ', loss)
       
    make_gif(filenames) 
