This is a simple and narrow hack to mask a fast wavelet transform as a
pseudo-matrix, so that generic matrix-multiplying code can be used for all
kernels (i.e. ones that can be implemented by efficient transforms as well as
ones that can be only implemented as matrices). Only functionality directly
needed by the sRVM code is currently implemented (only a few operations are
supported and these only for wavelets with a hard-coded wavelet package), but
the code could be easily generalized.

-- Alexander Schmolck, 2006-06-02