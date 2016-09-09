import numpy as np
from numpy.linalg import det, inv, norm
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

class GP:

    ############################################################################################
    # Initialize
    ############################################################################################
    def __init__(self, covariance_fn):
        self.cov   = covariance_fn
        self.X     = np.array([])
        self.y     = np.array([])
        self.K     = None

    ############################################################################################
    # Utilities
    ############################################################################################
    ###############################################################
    # Predict a new value
    # This shouldn't be its own function;
    # we do almost the same thing in mean_cov_at_new_points.
    # GPML, page 16
    ###############################################################
    def get_new_preds(self, new_point):
        # New point to predict
        # Only called when we have observed data!
        self.kinv = np.linalg.pinv(self.K)

        new_row   = [self.cov(new_point, xv) for xv in self.X]
        pred_mean = np.dot(np.dot(new_row, self.kinv), self.y)
        pred_2sds = 2 * np.sqrt(self.cov(new_point, new_point) - np.dot(np.dot(new_row, self.kinv), new_row))
        return pred_mean, pred_mean - pred_2sds, pred_mean + pred_2sds


    ###############################################################
    # Mean/Cov of GP conditioned on the observations
    # See GPML, page 16
    ###############################################################
    def mean_cov_at_new_points(self, locations):

        if self.K is not None:
            # We have some observations and we use them.
            # K* and K**
            self.kinv = np.linalg.pinv(self.K)
            ks        = np.array([[self.cov(l, xv) for xv in self.X] for l in locations])
            kss       = np.array([[self.cov(l, j) for l in locations] for j in locations])
            #print('ks', ks.shape)
            #print('kss', kss.shape)
            # The mean for the new point is:
            # (cov(x*, x).K^-1).y
            # Intuitively... decorrelate and then project onto y? That seems wrong.
            # What should the posterior mean be, if not this?
            # http://www.cs.cmu.edu/~16831-f14/notes/F09/lec21/16831_lecture21.sross.pdf
            pred_mean = np.dot(ks, np.dot(self.kinv, self.y))
            pred_cov  = np.subtract(kss, np.dot(ks, np.dot(self.kinv, np.transpose(ks))))

            return pred_mean, pred_cov
        else:
            # We have no observations, just sample from the prior:
            cov       = np.array([[self.cov(l, j) for l in locations] for j in locations])
            mean      = np.zeros(len(locations))
            return mean, cov

    ###############################################################
    # Generate samples from a multivariate Gaussian distribution:
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # or GPML, appendix A
    # (alternatively just use built-in function)
    ###############################################################
    def draw_from_dist(self, how_many, mean_vector, covariance_matrix):
        # find A such that A * A^T = covariance_matrix
        try:
            A = np.linalg.cholesky(covariance_matrix)
        except np.linalg.linalg.LinAlgError:
            A = sqrtm(covariance_matrix)

        z = np.random.normal(0, 1, (how_many, covariance_matrix.shape[0]))
        return np.array([np.add(mean_vector, np.dot(A, x)) for x in z])

    ##########################
    # Observe
    ##########################
    def observe(self, X, y, noise = False, sigma = .3):
        self.X = np.concatenate((self.X, X))
        self.y = np.concatenate((self.y, y))
        # Should be reworked to not recompute the whole thing every time:
        self.K = np.array([[self.cov(self.X[j], self.X[i]) for i in range(len(self.X))] for j in range(len(self.X))])
        if noise:
            # Page 16, eq 2.21
            self.K = self.K + sigma**2 * np.identity(len(self.K))

    def forget(self):
        self.X = np.array([]); self.y = np.array([]); self.K = None

    ##########################
    # Model assessment
    ##########################
    def model_log_likelihood(self):
        self.kinv = np.linalg.pinv(self.K)
        ys        = np.dot(self.K, np.dot(self.kinv, self.y))
        kdet      = (lambda x: x if x != 0 else .0001)(np.linalg.det(self.K))
        pred_ll   = -.5 * (np.dot(np.dot(ys, self.kinv), ys) + np.log(kdet) + len(self.y)*np.log(2*np.pi))
        return pred_ll

    ##########################
    # Sample
    ##########################
    def sample(self, locations, how_many_fns=1):
        self.sample_locations = locations
        mean, covariance      = self.mean_cov_at_new_points(self.sample_locations)
        self.values           = self.draw_from_dist(how_many_fns, mean, covariance)
        # Sanity check: does mine look like the built-in?
        #self.values = np.random.multivariate_normal(mean, covariance, how_many_fns)

    ##########################
    # Plot
    ##########################
    def plot(self):
        alpha = (.2 if self.K is not None else .8)

        for i in range(len(self.values)):
            plt.plot(self.sample_locations, self.values[i], alpha=alpha)

        if self.K is not None:
            vals = [self.get_new_preds(v) for v in self.sample_locations]
            plt.plot(self.sample_locations, [a for a, _, _ in vals], '-', color='k', alpha= 1)
            plt.plot(self.sample_locations, [a for _, a, _ in vals], '-', color='k', alpha=.5)
            plt.plot(self.sample_locations, [a for _, _, a in vals], '-', color='k', alpha=.5)

            # observed
            plt.plot(self.X, self.y, 'o', color = 'k');

def noisy_obs_title(model, sigma, ll):
    return model + ': Noisy Observations (SD of error = ' + str(sigma) + ')\nLog-likelihood: ' + str(ll)
def noise_free_title(model, ll):
    return model + ": Noise-Free Observations\nLog-likelihood: " + str(ll)
def saw_nothing_title(model):
    return model + " (prior)"
