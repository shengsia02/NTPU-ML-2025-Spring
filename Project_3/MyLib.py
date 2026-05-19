import numpy as np
import matplotlib.pyplot as plt

#### (1) 定義 show_Yale 函數。
# **使用說明：** \
# 將圖片排列成 montage 的形式，並輸出成圖片。

# **Inputs**: 
# - `X`: row 為 pixels，column 為樣本的影像矩陣。 
# - `m`, `n`: m $\times$ n 的圖片大小。 
# - `h`, `w`: 圖片的大小參數。

# **Output**: \
# 繪製出一張 figsize = (w, h) 的 $h \times w$ montage 圖。

def show_Yale(X, m, n, h, w):
    '''
    X: image matrix in which each column represents an image
    m, n: image size m x n
    h, w : create an h x w montage image with figsize = (w,h)
    '''
    fig, axes = plt.subplots(h, w, figsize=(w, h))
    if X.shape[1] < w * h: # 影像張數不到 w x h 張，用 0 向量補齊     
        X = np.c_[X, np.zeros((X.shape[0], w*h-X.shape[1]))]
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[:,i].reshape(m, n).T, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    # plt.suptitle(title, y=1.1)
    plt.show()

# ---------------------------------------

#### (2) 定義 show_AT 函數。
# **使用說明：** \
# 將圖片排列成 montage 的形式，並輸出成圖片。

# **Inputs**: 
# - `X`: row 為 pixels，column 為樣本的影像矩陣。 
# - `m`, `n`: m $\times$ n 的圖片大小。 
# - `h`, `w`: 圖片的大小參數。

# **Output**: \
# 繪製出一張 figsize = (w, h) 的 $h \times w$ montage 圖。

def show_AT(X, m, n, h, w):
    '''
    X: image matrix in which each column represents an image
    m, n: image size m x n
    h, w : create an h x w montage image with figsize = (w,h)
    '''
    fig, axes = plt.subplots(h, w, figsize=(w, h))
    if X.shape[1] < w * h: # 影像張數不到 w x h 張，用 0 向量補齊     
        X = np.c_[X, np.zeros((X.shape[0], w*h-X.shape[1]))]
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[:,i].reshape(m, n), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# ---------------------------------------

#### (3) 定義 format_runtime 函數。
def format_runtime(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"