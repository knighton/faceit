def max_pool_2d_nchw(x, k):
    if isinstance(k, int):
        kh = k
        kw = k
    else:
        kh, kw = k
    n, c, h, w = x.shape
    ih = h - h % kh
    iw = w - w % kw
    oh = h // kh
    ow = w // kw
    x = x[:, :, :ih, :iw]
    x = x.reshape(n, c, oh, kh, ow, kw)
    return x.max(axis=(3, 5))
