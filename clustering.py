import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans,KMeans,DBSCAN
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
# import hdbscan
from sklearn.metrics import rand_score, normalized_mutual_info_score
from collections import Counter, defaultdict
import ipdb

def assert_all_equal(lst):
    assert all(x == lst[0] for x in lst), "Not all elements in the list are equal."

def load_mapping(mapping_file):
    wnid_to_index = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            wnid, index = line.strip().split('\t')
            wnid_to_index[wnid] = int(index)
    return wnid_to_index

def load_latents_kl(mean_dir, logvar_dir, class_id):
    mean_files = sorted(os.listdir(os.path.join(mean_dir, class_id)))
    logvar_files = sorted(os.listdir(os.path.join(logvar_dir, class_id)))
    
    means = [torch.load(os.path.join(mean_dir, class_id, f), weights_only=True).cpu().numpy() for f in mean_files]
    logvars = [torch.load(os.path.join(logvar_dir, class_id, f), weights_only=True).cpu().numpy() for f in logvar_files]
    
    return np.array(means), np.array(logvars)


def kl_divergence(mean1, logvar1, mean2, logvar2, epsilon=1e-2):
    # Compute variances from log variances and add epsilon for stability
    var1 = np.exp(logvar1)
    var2 = np.exp(logvar2)
    
    # Clip variances to avoid extremely small or zero values
    var1 = np.clip(var1, epsilon, 1e10)
    var2 = np.clip(var2, epsilon, 1e10)
    
    # Compute KL divergence with added numerical stability
    try:
        kl = 0.5 * (np.sum(var1 / var2 + (mean2 - mean1)**2 / var2 - 1 + np.log(var2 / var1)))
    except FloatingPointError:
        kl = np.inf  # In case of a numerical issue, set KL to a high value
    
    # Check if the result is NaN or Inf, replace with a large number if so
    if np.isnan(kl) or np.isinf(kl):
        kl = 1e10  # Assign a large fallback value if divergence is invalid

    return kl

def cosine_similarity_batch(A, B):
    """
    Compute cosine similarity between two batches of vectors.
    Args:
        A: numpy array of shape (n_samples_1, n_features)
        B: numpy array of shape (n_samples_2, n_features)
    Returns:
        similarities: numpy array of shape (n_samples_1, n_samples_2)
    """
    # Normalize the vectors
    A_normalized = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_normalized = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarity
    return np.dot(A_normalized, B_normalized.T)

def compute_cosine_similarity_matrix(vectors):
    """
    Compute cosine similarity matrix efficiently.
    """
    # Normalize the vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / (norms + 1e-10)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    return similarity_matrix

def k_center_cos(class_latents, k, return_distances=False):
    """
    Perform K-center clustering on class latents using cosine similarity.
    
    Args:
        class_latents: numpy array of shape (n_samples, n_features)
        k: number of centers to select
        return_distances: whether to return minimum distances to centers
    
    Returns:
        centers: indices of selected centers
        assignments: cluster assignments for each point
        min_distances: (optional) minimum distances to centers for each point
    """

    # Initialize centers with a random point
    center_indices = [0]

    # Initialize similarity matrix
    similarities = cosine_similarity_batch(
        class_latents[center_indices], 
        class_latents
    )
        
    # Convert similarities to distances (1 - similarity)
    distances = 1 - similarities

    # Select remaining centers
    for _ in range(k - 1):
        # Find minimum distance to any existing center for each point
        min_distances = distances.min(axis=0)
        
        # Select the point with maximum minimum distance as the new center
        new_center = min_distances.argmax()
        center_indices.append(new_center)
        
        # Update distances matrix with the new center
        new_similarities = cosine_similarity_batch(
            class_latents[[new_center]], 
            class_latents
        )
        new_distances = 1 - new_similarities
        distances = np.vstack([distances, new_distances])
    
    # Compute final assignments
    assignments = distances.argmin(axis=0)
    
    if return_distances:
        min_distances = distances.min(axis=0)
        return center_indices, assignments, min_distances
    
    return center_indices, assignments


def k_center_kl(means, logvars, K):
    n = len(means)
    selected = [0]  # Start with the first point as the first center
    distances = np.full(n, np.inf)  # Initialize distances with infinity
    
    for _ in range(1, K):
        # Update distances to the nearest selected center
        for i in range(n):
            d = kl_divergence(means[selected[-1]], logvars[selected[-1]], means[i], logvars[i])
            distances[i] = min(distances[i], d)
        
        # Choose the next center as the point with the maximum distance to its nearest center
        next_center = np.argmax(distances)
        selected.append(next_center)
    
    return selected


def visualize_latents_kl(means, selected_indices, method='tsne'):
    if method == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        reducer = PCA(n_components=2)
    
    reduced_latents = reducer.fit_transform(means)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_latents[:, 0], reduced_latents[:, 1], c='gray', label='Latents')
    plt.scatter(reduced_latents[selected_indices, 0], reduced_latents[selected_indices, 1], 
                c='red', label='Selected Centers', edgecolors='black')
    plt.title(f'Visualization of Latent Space with {len(selected_indices)} Centers')
    plt.legend()
    plt.show()



def visualize_clusters_cos(class_latents, center_indices, assignments, class_name, save_dir=None):
    """
    Create comprehensive visualization of clustering results.
    
    Args:
        class_latents: numpy array of shape (n_samples, n_features)
        center_indices: list of indices of cluster centers
        assignments: numpy array of cluster assignments
        class_name: string name of the class
        save_dir: directory to save the plots (optional)
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. t-SNE visualization
    ax1 = fig.add_subplot(gs[0, 0])
    tsne = TSNE(n_components=2, random_state=42)
    embedded_latents = tsne.fit_transform(class_latents)
    
    # Plot all points
    scatter = ax1.scatter(embedded_latents[:, 0], embedded_latents[:, 1], 
                         c=assignments, cmap='tab20', alpha=0.6, s=50)
    
    # Highlight centers
    ax1.scatter(embedded_latents[center_indices, 0], embedded_latents[center_indices, 1],
                c='red', marker='*', s=200, label='Centers')
    
    ax1.set_title(f't-SNE Visualization\n{class_name}')
    ax1.legend()
    
    # 2. PCA visualization
    ax2 = fig.add_subplot(gs[0, 1])
    pca = PCA(n_components=2)
    embedded_pca = pca.fit_transform(class_latents)
    
    # Plot all points
    scatter_pca = ax2.scatter(embedded_pca[:, 0], embedded_pca[:, 1],
                             c=assignments, cmap='tab20', alpha=0.6, s=50)
    
    # Highlight centers
    ax2.scatter(embedded_pca[center_indices, 0], embedded_pca[center_indices, 1],
                c='red', marker='*', s=200, label='Centers')
    
    ax2.set_title(f'PCA Visualization\n{class_name}')
    ax2.legend()
    
    # 3. Cluster size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    cluster_sizes = np.bincount(assignments)
    centers = np.arange(len(cluster_sizes))
    ax3.bar(centers, cluster_sizes)
    ax3.set_title('Cluster Size Distribution')
    ax3.set_xlabel('Cluster Index')
    ax3.set_ylabel('Number of Samples')
    
    # 4. Distance matrix visualization
    # ax4 = fig.add_subplot(gs[1, 1])
    # # Compute pairwise cosine similarities between centers
    # center_latents = class_latents[center_indices]
    # center_latents_norm = center_latents / (np.linalg.norm(center_latents, axis=1, keepdims=True) + 1e-10)
    # similarities = np.dot(center_latents_norm, center_latents_norm.T)
    
    # # Plot similarity matrix
    # im = ax4.imshow(similarities, cmap='viridis')
    # plt.colorbar(im, ax=ax4)
    # ax4.set_title('Center-to-Center Similarity Matrix')
    # ax4.set_xticks(range(len(center_indices)))
    # ax4.set_yticks(range(len(center_indices)))
    
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{class_name}_clustering.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.close()

def visualize_cos_similarities(class_latents, center_indices, assignments, class_name, save_dir=None):
    """
    Create visualization of cosine similarities and clustering results.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Full cosine similarity matrix
    ax1 = fig.add_subplot(gs[0, 0])
    similarity_matrix = compute_cosine_similarity_matrix(class_latents)
    
    im1 = ax1.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(im1, ax=ax1, label='Cosine Similarity')
    ax1.set_title('Full Cosine Similarity Matrix')
    
    # Add cluster boundaries if number of samples is not too large
    if len(class_latents) < 500:  # Only add lines for smaller datasets
        # Sort by cluster assignment
        sort_idx = np.argsort(assignments)
        sorted_sim = similarity_matrix[sort_idx][:, sort_idx]
        
        # Get cluster boundaries
        boundaries = np.where(np.diff(assignments[sort_idx]))[0]
        
        # Add lines to show cluster boundaries
        for b in boundaries:
            ax1.axhline(y=b+0.5, color='black', linestyle='-', alpha=0.3)
            ax1.axvline(x=b+0.5, color='black', linestyle='-', alpha=0.3)
    
    # 2. Center-to-all points similarity
    ax2 = fig.add_subplot(gs[0, 1])
    center_latents = class_latents[center_indices]
    center_similarities = compute_cosine_similarity_matrix(center_latents)
    
    im2 = ax2.imshow(center_similarities, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(im2, ax=ax2, label='Cosine Similarity')
    ax2.set_title('Center-to-Center Similarity Matrix')
    
    # 3. Similarity distribution histogram
    ax3 = fig.add_subplot(gs[1, 0])
    # Get upper triangle values (excluding diagonal)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[triu_indices]
    
    ax3.hist(similarities, bins=50, density=True)
    ax3.set_title('Distribution of Cosine Similarities')
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    
    # Add vertical lines for mean and median
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    ax3.axvline(mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    ax3.axvline(median_sim, color='g', linestyle='--', label=f'Median: {median_sim:.3f}')
    ax3.legend()
    
    # 4. Within vs Between cluster similarities
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Compute within and between cluster similarities
    within_sims = []
    between_sims = []
    
    for i in range(len(class_latents)):
        for j in range(i+1, len(class_latents)):
            sim = similarity_matrix[i, j]
            if assignments[i] == assignments[j]:
                within_sims.append(sim)
            else:
                between_sims.append(sim)
    
    # Plot as boxplots
    box_data = [within_sims, between_sims]
    ax4.boxplot(box_data, labels=['Within Cluster', 'Between Clusters'])
    ax4.set_title('Within vs Between Cluster Similarities')
    ax4.set_ylabel('Cosine Similarity')
    
    # Add means as points
    means = [np.mean(within_sims), np.mean(between_sims)]
    ax4.scatter([1, 2], means, color='red', marker='*', s=100, label='Mean')
    ax4.legend()
    
    # Overall title
    plt.suptitle(f'Similarity Analysis for Class: {class_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{class_name}_similarities.png'), 
                    dpi=300, bbox_inches='tight')
    
    plt.close()


def visualize_clusters(latents, cluster_labels, save_dir=None):
    """
    Visualize clustering results using PCA and t-SNE in both 2D and 3D.
    
    Args:
        latents: numpy array of shape (n_samples, n_features)
        cluster_labels: numpy array of cluster assignments
        save_dir: directory to save plots (optional)
    """
    # Set style
    plt.style.use('default')  # Using default style instead of seaborn
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Color palette
    n_clusters = len(np.unique(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))  # Using matplotlib colormap
    
    # 1. PCA visualization (2D)
    ax1 = fig.add_subplot(231)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latents)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=[colors[i]], label=f'Cluster {i}',
                   alpha=0.6)
    
    ax1.set_title('PCA 2D Visualization')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.legend()
    ax1.grid(True)
    
    # 2. PCA visualization (3D)
    ax2 = fig.add_subplot(232, projection='3d')
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(latents)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2],
                   c=[colors[i]], label=f'Cluster {i}',
                   alpha=0.6)
    
    ax2.set_title('PCA 3D Visualization')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.grid(True)
    
    # 3. t-SNE visualization (2D)
    ax3 = fig.add_subplot(234)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(latents)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax3.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}',
                   alpha=0.6)
    
    ax3.set_title('t-SNE 2D Visualization')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.legend()
    ax3.grid(True)
    
    # 4. t-SNE visualization (3D)
    ax4 = fig.add_subplot(235, projection='3d')
    tsne = TSNE(n_components=3, random_state=42)
    tsne_result = tsne.fit_transform(latents)
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax4.scatter(tsne_result[mask, 0], tsne_result[mask, 1], tsne_result[mask, 2],
                   c=[colors[i]], label=f'Cluster {i}',
                   alpha=0.6)
    
    ax4.set_title('t-SNE 3D Visualization')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.set_zlabel('t-SNE 3')
    ax4.grid(True)
    
    # 5. Cluster size distribution
    ax5 = fig.add_subplot(233)
    cluster_sizes = np.bincount(cluster_labels)
    ax5.bar(range(n_clusters), cluster_sizes, color=colors)
    ax5.set_title('Cluster Size Distribution')
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Number of Samples')
    ax5.grid(True)
    
    # Add explained variance ratio for PCA
    pca = PCA()
    pca.fit(latents)
    ax6 = fig.add_subplot(236)
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    ax6.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, 'bo-')
    ax6.set_title('PCA Explained Variance Ratio')
    ax6.set_xlabel('Number of Components')
    ax6.set_ylabel('Cumulative Explained Variance Ratio')
    ax6.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/5_class_cluster_visualization.png', dpi=300, bbox_inches='tight')
    
    plt.show()

#-------------------------------- Perform K-center clustering KL DIVERGANCE ---------------------------
# K = 10
# mean_dir = '/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/vae_distribution/mean'
# logvar_dir = '/home/user1/workspace/leilu/linzhao/dataset/tiny-imagenet-200/vae_distribution/logvar'
# classes = sorted(os.listdir(mean_dir))
# for class_name in classes:
#     means, logvars = load_latents_kl(mean_dir, logvar_dir, class_name)
    
#     # Perform K-center algorithm
#     selected_indices = k_center_kl(means, logvars, K)
    
#     # Visualize the result
#     visualize_latents_kl(means, selected_indices, method='tsne')
#------------------------------------------------------------------------------------------------------

def vae():
    #-------------------------------- Perform K-means clustering to 5 classes ---------------------------
    wnid_to_index = load_mapping(os.path.join('../dataset/tiny-imagenet-200', 'tiny-imagenet-mapping.txt'))
    input_dir = '../dataset/tiny-imagenet-200/train_CLIP_image_feature'
    key_latent_dir = '../dataset/tiny-imagenet-200/key10-real_mindis_latents_train_ddim_inversion_s20_kmeasns++'
    n_clusters = 10
    # class_folders = ['n01443537']#, 'n01629819', 'n01641577', 'n01644900', 'n01698640']  
    # latents = []
    # labels = []
    for class_idx, class_folder in enumerate(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_folder)
        latents = []
        labels = []
        if os.path.isdir(class_dir):
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_dir, file_name)
                    latent = torch.load(file_path, weights_only=True)  # Load the latent .pth file
                    latents.append(latent.cpu().numpy())
                    labels.append(wnid_to_index[file_name.split('_')[0]])
            assert_all_equal(labels)  #

            original_shape = latents[0].shape  # 保存原始的形状
            latents = np.stack(latents).astype(np.float32)
            class_latents = latents.reshape(latents.shape[0], -1)
            
            #--------------- kmeans++ ----------------------------------------------------------------
            
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
            kmeans.fit(class_latents)
            y_kmeans = kmeans.predict(class_latents)
            
            # 获取每个簇的真实样本，即距离簇中心最近的样本
            key_latents = []
            for cluster_idx in range(n_clusters):
                # 找出属于当前簇的样本
                cluster_samples = class_latents[y_kmeans == cluster_idx]
                
                # 计算这些样本到当前簇中心的距离
                cluster_center = kmeans.cluster_centers_[cluster_idx]
                distances = np.linalg.norm(cluster_samples - cluster_center, axis=1)
                
                # 找到距离簇中心最近的样本
                closest_sample = cluster_samples[np.argmin(distances)]
                key_latents.append(closest_sample)
            
            # key_latents = kmeans.cluster_centers_
            key_latents = np.array(key_latents)
            key_latents_4d = key_latents.reshape(n_clusters, *original_shape)
            number = np.bincount(y_kmeans)
        
            
            #--------------- minibatch [LATENT], 后续考虑换成kmeans，因为更精准，数据量小 --------------------------
            '''
            minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=0)
            minibatch_kmeans.fit(class_latents)
            y_kmeans = minibatch_kmeans.predict(class_latents)
            
            key_latents = minibatch_kmeans.cluster_centers_
            key_latents_4d = key_latents.reshape(n_clusters, *original_shape)
            number = np.bincount(y_kmeans)
            '''
            
            #----------------- HDBSCAN for imbalanced dataset [NOISE] 不行，都被聚类成了-1（noise）----------------
            '''
            # 归一化特征
            scaler = StandardScaler()
            class_latents = scaler.fit_transform(class_latents)

            # 使用 HDBSCAN 进行聚类
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, #int(len(class_latents) / n_clusters),
                                        min_samples=1,  # 根据数据调整
                                        cluster_selection_epsilon=0.5)  # 通过 epsilon 控制簇的生成
            clusterer.fit(class_latents)
            y_hdbscan = clusterer.labels_  # 聚类标签

            # 计算每个类的中心 latent
            unique_labels = np.unique(y_hdbscan[y_hdbscan != -1])  # 排除噪声
            key_latents = np.array([class_latents[y_hdbscan == label].mean(axis=0) for label in unique_labels])

            # 恢复成原始形状
            key_latents_4d = key_latents.reshape(len(unique_labels), *original_shape)
            number = [np.sum(y_hdbscan == label) for label in unique_labels]
            '''
            #--------------------------------------------------------------------------------------

            # 保存每个关键 latent 到单独的 .pt 文件
            class_save_dir = os.path.join(key_latent_dir, class_dir.split('/')[-1])
            os.makedirs(class_save_dir, exist_ok=True)
            
            for i, key_latent in enumerate(key_latents_4d):
                torch.save(torch.tensor(key_latent), os.path.join(class_save_dir, f'key_latent_{i}_{number[i]}.pt'))

            

    # 使用 MiniBatchKMeans 进行聚类
    # n_clusters = 5
    # minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=0)
    # minibatch_kmeans.fit(class_latents)
    # y_kmeans = minibatch_kmeans.predict(class_latents)

    # # Perform K-means clustering
    # num_clusters = 10
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(class_latents)

    # visualize_clusters(
    #         latents=class_latents,
    #         cluster_labels=cluster_labels,
    #         save_dir='clustering_visualizations'
    # )
    # #------------------------------------------------------------------------------------------------------


    '''
    wnid_to_index = load_mapping(os.path.join('../dataset/tiny-imagenet-200', 'tiny-imagenet-mapping.txt'))
    # Latents 文件夹路径
    key_latent_dir = '../dataset/tiny-imagenet-200/key10-latents_train_vae_encode'
    os.makedirs(key_latent_dir, exist_ok=True)


    input_dir = '../dataset/tiny-imagenet-200/train_ddim_inversion_s20'
    n_clusters = 10  # 每个类别聚类的中心数

    for class_name in sorted(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_name)
        # 存储所有的 latent 表示
        latents = []
        labels = []

        if os.path.isdir(class_dir):
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_dir, file_name)
                    latent = torch.load(file_path, weights_only=True)  # Load the latent .pth file
                    latents.append(latent.cpu().numpy())
                    labels.append(wnid_to_index[file_name.split('_')[0]])
            assert_all_equal(labels)  #

            original_shape = latents[0].shape  # 保存原始的形状
            # 将 latents 转换为 numpy 数组
            latents = np.stack(latents).astype(np.float32)
            # 如果 latents 是 4 维数据，将其展平成 2 维 (num_samples, -1)
            class_latents = latents.reshape(latents.shape[0], -1)


            #-------------------------------- Perform K-center clustering COSINE ----------------------------------
            # k = max(1, int(0.1 * len(class_latents)))  # Use 10% of samples as centers, adjust as needed
            k = 10
            center_indices, assignments = k_center_cos(class_latents, k)

            save_dir = 'clustering_visualizations_noise'
            visualize_cos_similarities(class_latents, center_indices, assignments, 
                                class_name, save_dir=save_dir)
            # visualize_clusters(class_latents, center_indices, assignments, 
            #                  class_name, save_dir=save_dir)
            
            # Store or process the results as needed
            cluster_sizes = np.bincount(assignments)
            print(f"Class {class_name}: {k} centers, cluster sizes: {cluster_sizes}")
            
            # Optional: Store the centers and assignments
            # class_results = {
            #     'centers': center_indices,
            #     'assignments': assignments,
            #     'original_shape': original_shape,
            #     'cluster_sizes': cluster_sizes
            # }
            #------------------------------------------------------------------------------------------------------



            
            #-------------------------------- Perform K-center clustering KL DIVERGANCE ---------------------------
            # # 假设你有N个高斯分布, 每个有均值mu和协方差矩阵sigma
            # N = len(gaussian_istributions)
            # distance_matrix = np.zeros((N, N))

            # for i in range(N):
            #     for j in range(i+1, N):
            #         mu_p, sigma_p = gaussian_distributions[i]
            #         mu_q, sigma_q = gaussian_distributions[j]
            #         kl_ij = kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q)
            #         distance_matrix[i][j] = kl_ij
            #         distance_matrix[j][i] = kl_ij  # 对称矩阵
            
            # # 假设你希望得到K个中心
            # K = 3
            # centers, clusters = k_center_clustering(distance_matrix, K)
            #------------------------------------------------------------------------------------------------------



            # 使用 MiniBatchKMeans 进行聚类
            # minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=0)
            # minibatch_kmeans.fit(class_latents)
            # y_kmeans = minibatch_kmeans.predict(class_latents)
            
            # kmeans = KMeans(n_clusters=n_clusters)
            # kmeans.fit(class_latents)
            # y_kmeans = kmeans.predict(class_latents)
            
            # 执行 DBSCAN 聚类
            # dbscan = DBSCAN(eps=0.005, min_samples=20)
            # labels_p = dbscan.fit_predict(class_latents)
            
            
            # 可视化 PCA 降维后的聚类结果
            # pca = PCA(n_components=2)
            # X_pca = pca.fit_transform(class_latents)

            # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_p, s=1, cmap='viridis')
            # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=1, cmap='viridis')

            # 添加标题和坐标轴标签
            # plt.title('K-means Clustering with PCA 2D Visualization')
            # plt.xlabel('PCA Component 1')
            # plt.ylabel('PCA Component 2')

            
            # # 使用 t-SNE 将高维数据降到 2 维
            # tsne = TSNE(n_components=2, random_state=0)
            # X_tsne = tsne.fit_transform(class_latents)

            # # 可视化降维后的聚类结果
            # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, s=1, cmap='viridis')

            # # t-SNE 不保留全局结构，不适合降维后的聚类中心展示
            # # 所以这里我们不绘制聚类中心

            # plt.title('K-means Clustering with t-SNE Visualization')
            # plt.xlabel('t-SNE Component 1')
            # plt.ylabel('t-SNE Component 2')
            # plt.savefig('minibatch_10_vis.png')
            # plt.show()
            
            # 获取每个类别的聚类中心（关键 latent）
            # key_latents = minibatch_kmeans.cluster_centers_
            # # 将 key_latents 还原为 4 维 (n_clusters, C, H, W)
            # key_latents_4d = key_latents.reshape(n_clusters, *original_shape)

            # # 保存每个关键 latent 到单独的 .pt 文件
            # class_save_dir = os.path.join(key_latent_dir, class_dir.split('/')[-1])
            # os.makedirs(class_save_dir, exist_ok=True)
            
            # for i, key_latent in enumerate(key_latents_4d):
            #     torch.save(torch.tensor(key_latent), os.path.join(class_save_dir, f'key_latent_{i}.pt'))

            # print(f"Class {labels[0]}: Saved {n_clusters} key latents.")
    '''


# --------------------------------------------------- CLIP -----------------------------------------------------
# --------------------------------------------------- ViT -----------------------------------------------------

def get_sorted_space(x, y, z):
    init_count = 0
    for count in x:
        count = count + init_count
        y.append(z[init_count:count])
        init_count = count


def vit_clustering():
    #-------------------------------- Perform K-means++ clustering to 200 classes ---------------------------
    wnid_to_index = load_mapping(os.path.join('ds_inf/tiny-imagenet-200', 'tiny-imagenet-mapping.txt'))
    input_dir = '/scratch/zhao.lin1/dataset/tiny-imagenet-200/train_vit_image_feature_CLS'
    save_dir = '/scratch/zhao.lin1/dataset/tiny-imagenet-200/key50_far_latents_train_vit_image_feature_CLS'
    n_clusters = 200
    n_key = 50
    latents = []
    labels = []
    file_paths = []
    for class_idx, class_folder in enumerate(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_folder)
        # latents = []
        # labels = []
        if os.path.isdir(class_dir):
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_dir, file_name)
                    file_paths.append(file_path)
                    latent = torch.load(file_path, weights_only=True)  
                    latents.append(latent.cpu().numpy())
                    labels.append(wnid_to_index[file_name.split('_')[0]])
                    # labels.append(class_idx)

    class_latents = np.stack(latents).astype(np.float32)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
    kmeans.fit(class_latents)
    y_kmeans = kmeans.predict(class_latents)
    mapping_candidates = defaultdict(list)
    
    
    
    subspace_information_list, subspace_list, counter = [], [], []
    label_basis, sorted_subspace = [], []
    farest_sort, cluster_distance_space = [], []
    distance_space = kmeans.fit_transform(
            class_latents
        )

    # stats, conf_matrix = analyze_clip_clustering(save_dir='clip_clustering', y_kmeans=y_kmeans, labels=labels)
    # print(stats)
    true_labels = np.array(labels)
    predicted_clusters = np.array(y_kmeans)
    
    
    for sample, true in zip(predicted_clusters, true_labels):
        mapping_candidates[sample].append(true)

    # 使用 Counter 找出每个 sample_label 的最常见的 true_label
    label_mapping = {sample: Counter(trues).most_common(1)[0][0]
                    for sample, trues in mapping_candidates.items()}
    
    
    distance_space_information = [
        [sub_space_index,np.argmin(distance_subspace),
         np.min(distance_subspace)]
        for sub_space_index, distance_subspace in enumerate(distance_space)
    ]

    number_cluster = len(np.unique(
        [i[1] for i in distance_space_information])
    )

    for label in range(number_cluster):
        count = 0
        for index, distance_subspace_information in \
                enumerate(distance_space_information):
            if distance_subspace_information[1] == label:
                count += 1
                subspace_list.append(
                    distance_subspace_information
                )
        counter.append(count)

    # get data subspace information on label basis
    get_sorted_space(
        counter,
        subspace_information_list,
        subspace_list
    )

    for subspace_information in subspace_information_list:
        for index, subspace in enumerate(subspace_information):
            label_basis.append(subspace[2])

    #  get the distance subspace information based on labels
    get_sorted_space(
        counter,
        sorted_subspace,
        label_basis
    )

    # get sorted subspace index based on farest prototype
    for i in sorted_subspace:
        farest_sort.append(np.argsort(i)[::-1])

    # get the sort files base on subspace information
    for i in range(len(farest_sort)):
        cluster_distance_space.append(
            [subspace_information_list[i][v][0]
             for j, v in enumerate(farest_sort[i])]
        )
        
    json_data = {}
        
    for i in range(number_cluster):
        num = 0
        
        for k in range(len(cluster_distance_space[i])):
            id = cluster_distance_space[i][k]
            if label_mapping[predicted_clusters[id]] == true_labels[id]:
                key = file_paths[id].split('/')[-2]
                value = file_paths[id].split('/')[-1].replace('.pt', '.JPEG')
                
                if key not in json_data:
                    json_data[key] = []
                json_data[key].append(value)
            
                # torch.save(torch.tensor(class_latents[id], dtype=torch.float16), os.path.join(save_dir, file_paths[id].split('/')[-2], file_paths[id].split('/')[-1]))
                
                num = num+1
            if num >= n_key:
                break
                
                
    with open(os.path.join('ds_inf','key50_far_latents_train_vit_image_feature_CLS.json'), 'w') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
                
    
    
    # # Calculate clustering metrics
    # ari = rand_score(true_labels, predicted_clusters)
    # nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
    # print(f"Adjusted Rand Index: {ari:.3f}")
    # print(f"nmi: {nmi:.3f}")
    
    # # Reduce dimensionality for visualization
    # pca = PCA(n_components=50)
    # latents_pca = pca.fit_transform(class_latents)
    # tsne = TSNE(n_components=2, random_state=0)
    # latents_2d = tsne.fit_transform(latents_pca)

    # # Plot the results
    # plt.figure(figsize=(10, 5))

    # # Plot original labels
    # plt.subplot(1, 2, 1)
    # scatter1 = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab20', marker='o', alpha=0.7, s=1)
    # plt.colorbar(scatter1, label='Original Class Labels')
    # plt.title('Original Class Labels')

    # # Plot cluster labels from K-means++
    # plt.subplot(1, 2, 2)
    # scatter2 = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=y_kmeans, cmap='tab10', marker='o', alpha=0.7,s=1)
    # plt.colorbar(scatter2, label='K-means Cluster Labels')
    # plt.title('K-means++ Cluster Labels')
    # plt.savefig('200class_clustering/cos_Full_TinyImgNet_VIT_200_Kmeans++.png', dpi=300, bbox_inches='tight')
    
    # plt.show()



 
                
                
def main():
    vit_clustering()
    
if __name__ == "__main__":
    main()

            