import numpy as np
import matplotlib.pyplot as plt

#### (1) 定義 montage 函數。
# **使用說明：** \
# 將圖片排列成 montage 的形式，只限於**正方形**的圖片，例如 28 $\times$ 28 的圖片。

# **Inputs**: 
# - `A`: 原始圖片的資料，shape = $(p, N)$，其中 $p$ = 每張圖片的像素數，$N$ = 圖片數量。 
# - `m`: montage 的 rows 數，$mn < N$。 
# - `n`: montage 的 columns 數。

# **Output**: \
# Montage 的圖片矩陣，shape = ($m \times sz$, $n \times sz$)，其中 $sz = \sqrt{p}$，以上面例子來說 $sz = 28$。

def montage(A, m, n):
    """
    將圖片排列成 montage 的形式, 只限於正方形的圖片
    例如: 28*28 的圖片, sz = 28
    Inputs:
    A: 原始的圖片資料, shape = (p, N), p = 每張圖片的像素數, N = 圖片數量
    m: montage 的 rows 數, m*n < N
    n: montage 的 columns 數
    Output:
    Montage 的圖片矩陣, shape = (m*sz, n*sz), sz = sqrt(p)
    """
    sz = np.sqrt(A.shape[0]).astype("int")
    M = np.zeros((sz*m, sz*n))

    for i in range(m*n):
        row = (i // n) * sz # // 代表整數除法
        col = (i % n) * sz  # % 代表取餘數
        M[row:row+sz, col:col+sz] = A[:, i].reshape(sz, sz)
    return M

# ---------------------------------------

#### (2) 定義 show_montage 函數。
# **使用說明：** \
# 將圖片排列成 montage 的形式，並輸出成圖片。

# **Inputs**: 
# - `X`: row 為 pixels，column 為樣本的影像矩陣。 
# - `m`, `n`: m $\times$ n 的圖片大小。 
# - `h`, `w`: 圖片的大小參數。

# **Output**: \
# 繪製出一張 figsize = (w, h) 的 $h \times w$ montage 圖。

def show_montage(X, m, n, h, w, title):
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
    plt.suptitle(title, y=1.1)
    plt.show()