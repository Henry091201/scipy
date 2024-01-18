__all__ = ['mycython']

cpdef int mycython():
    cdef int i = 1
    while i < 10000000:
        i += 1
    return i