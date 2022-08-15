import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise_distances
from sklearn.cluster import KMeans


def normIt(data, m=None):
    nData = data.copy()
    #nData = data/np.linalg.norm(data, axis =1, keepdims=True)
    if m is None:
        m = np.mean(nData, axis =0, keepdims=True)
    nData = nData - m
    nData = nData/np.linalg.norm(nData, axis =1, keepdims=True)
    
    return nData, m

def percentile_est(d):
    steps = np.array(range(100))
    percentiles = np.percentile(d, steps)
    return percentiles


class LabelStats():
    def __init__(self, mean=None, test_vec=None, md=None, percentiles=None):
        self.mean = mean
        self.test_vec = test_vec
        self.md = md
        self.percentiles = percentiles

    def estimate_from_feat(self, feat, test_vec):
        d2test = square_dist2_vec(feat, test_vec)
        m = np.mean(feat, axis=0, keepdims=True)
        d = pairwise_distances(feat, m)**2 - d2test
        percentiles = percentile_est(d)

        self.mean = m
        self.test_vec = test_vec
        self.percentiles = percentiles

    def est_percentile(self, feat):
        d2test = square_dist2_vec(feat, self.test_vec)
        dist = pairwise_distances(feat, self.mean)**2 - d2test[:,None]
        a = np.abs(self.percentiles - dist)
        per = np.argmin(a, axis=1)/100.0
        return per    




class DistanceClassifier():

    def __init__(self, 
                test_vec = None,
                num_clusters = 30, 
                num_reps =10):

        self.test_vec = test_vec
        self.num_clusters = num_clusters
        self.num_reps = num_reps
    
        self.centers = None
        self.center2labels = None
        self.stats = None
        self.num_class = 0
    
    def fit(self, feat, gt):

        num_class = np.max(gt) + 1
        class_index = np.unique(gt)
        print('Clusters per class:', self.num_clusters)  
        print('Repetitions per class:', self.num_reps)
        print('Number of classes:', num_class)

        print('processing class:', end =' ')
        for i in class_index:
            print(i, end = ', ')
            feat_ = feat[gt==i]
            centers, stats = one_class(feat_, self.num_clusters, self.num_reps, self.test_vec)
            labels = i * np.ones(centers.shape[0], dtype=int)
            self.add_center(centers, labels, [stats])
        print('\n')


    
    def add_center(self, centers, center2labels, stats=None):
        if self.num_class == 0:
            self.centers = centers
            self.center2labels = center2labels
            self.stats = stats
            self.num_class = np.max(center2labels) + 1
            return
        
        self.centers = np.concatenate([self.centers, centers], axis=0)
        self.center2labels = np.concatenate([self.center2labels, center2labels])
        self.num_class = np.max(center2labels) + 1
        
        new_labels = np.unique(center2labels)
        if stats is None:
            for i in new_labels:
                new_stats = self.stats[-1].copy()
                self.stats.append(new_stats)
        else:
            self.stats = self.stats + stats




    def predict(self, test_feat):
        dist = pairwise_distances(test_feat, self.centers)**2
        label = np.argsort(dist, axis=1)
        return self.center2labels[label]

    def decision_function(self, test_feat, use_test_vec):
        num_class = max(self.center2labels) + 1
        dist = pairwise_distances(test_feat, self.centers)**2


        d_ratio1 = 1 + dist / np.max(dist, axis=1, keepdims=True)
        
        d_ratio2 =  dist / np.min(dist, axis=1, keepdims=True)

        percentiles = np.zeros([test_feat.shape[0], num_class])

        if use_test_vec:
            print('full validation')
            if self.test_vec is None:
                print('Full validation requires test_vec to be initialized.')
                print('If test_vec cannot be estimated, set use_test_vec to false.')
            for i in range(num_class):
                p1 = np.min(d_ratio1[:,self.center2labels==i], axis=1)
                p2 = np.min(d_ratio2[:,self.center2labels==i], axis=1)
                p3 = self.stats[i].est_percentile(test_feat)
                percentiles[:,i] = (p1) * (p2) * (p3)
        else:
            print('ratio validation')
            for i in range(num_class):
                p1 = np.min(d_ratio1[:,self.center2labels==i], axis=1)
                p2 = np.min(d_ratio2[:,self.center2labels==i], axis=1)
                percentiles[:,i] = (p1) * (p2) 

        return - percentiles



    
def one_class(feat, k, r, test_vec=None):
    centers, _ = cluster_des_rep(feat, num_clus=k, num_reps=r)
    stats = LabelStats()    
    if test_vec is not None:
        stats.estimate_from_feat(feat, test_vec)
    return centers, stats


def square_dist2_vec(feat, test_vec, md=None):
    return np.linalg.norm(feat - test_vec, axis=1)**2


def cluster_des(class_feat, num_clus=30, normalize=True):
    mini_center = []
    if normalize:
        feat, _ = normIt(class_feat)
    else:
        feat = class_feat
        
    k = min(feat.shape[0]//10, num_clus)
    center2feat = np.zeros([feat.shape[0],k], dtype=bool)
    kmeans = KMeans(n_clusters=k, random_state=None).fit(feat)
    for j in range(k):
        mask = kmeans.labels_==j
        mini_center.append(np.mean(class_feat[mask], axis=0))
        center2feat[mask,j] = True
    return mini_center, center2feat
        
def cluster_des_rep(class_feat, num_clus=30, num_reps=10,  normalize=True):
    all_centers = []
    all_centers2feat = []

    for _ in range(num_reps):
        centers, center2feat = cluster_des(class_feat,num_clus, normalize)

        all_centers = all_centers + centers
        all_centers2feat.append(center2feat)
    
    all_centers = np.array(all_centers)
    all_centers2feat = np.concatenate(all_centers2feat, axis=1)

    return all_centers, all_centers2feat



#################### basic eval functions ###################


def auroc_eval(scores, gt, verboise=True):
    roc = []
    for i, score in enumerate(scores):
        roc.append(roc_auc_score(gt == i, score))

    auroc = np.mean(roc)
    if verboise:
        print('auroc:', auroc)
    return roc, auroc

def auprc_eval(scores, gt, verboise=True):
    prc = []
    for i, score in enumerate(scores):
        prc.append(average_precision_score(gt == i, score))

    auprc = np.mean(prc)
    if verboise:
        print('auprc:', auprc)
    return prc, auprc

    

