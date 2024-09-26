from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def predict_label(new_order_center, labeled_centers, labels, k=5):
    """
    预测新订单所属的标签
    """
    # 初始化KNeighborsClassifier模型
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # 使用已标记的地址和对应的标签训练模型
    knn.fit(labeled_centers, labels)
    
    # 预测新订单所属的标签
    predicted_label = knn.predict([new_order_center])
    
    return predicted_label[0]

# 示例数据
# 新订单中心地址
new_order_center = [40.7484, -73.9857]

# 一组已有标签的地址
labeled_centers = np.array([
    [40.7484, -73.9857],  # 示例已标记地址1
    [37.7749, -122.4194],  # 示例已标记地址2
    [34.0522, -118.2437],  # 示例已标记地址3
    # 添加更多已标记地址...
])

# 对应的标签
labels = np.array([1, 2, 3])  # 示例标签，与已标记地址一一对应

# 预测新订单所属的标签
predicted_label = predict_label(new_order_center, labeled_centers, labels)

print("新订单所属的标签:", predicted_label)