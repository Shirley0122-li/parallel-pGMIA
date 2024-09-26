from sklearn.cluster import KMeans
import numpy as np

def compute_center_points(locations):
    """
    Compute the centor points of each order
    """
    num_orders = locations.shape[0]
    center_points = np.zeros((num_orders, 2))
    for i in range(num_orders):
        center_points[i, 0] = (locations[i, 0] + locations[i, 2]) / 2  # 取餐地点的经度
        center_points[i, 1] = (locations[i, 1] + locations[i, 3]) / 2  # 取餐地点的纬度
    return center_points

def k_means(data, n_clusters):
    # the format of data is (414, 2) where 
    # first, second columes are latitude and longtitude of pick-up address
    # third, fourth columes are latitude and longtitude of customer address
    
    # the latitude and longtitude of each order is defined as 
    # the middle location of PickUp address and Customer address
    order_center_points = compute_center_points(data)

    # Initiilize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters)

    # Start clustering
    kmeans.fit(order_center_points)

    # obtain the label of cluster for each order
    # the shape of labels is 1d np array
    labels = kmeans.labels_

    return labels
