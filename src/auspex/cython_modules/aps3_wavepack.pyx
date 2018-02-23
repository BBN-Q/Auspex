import numpy as np
cimport numpy as np

cdef unsigned int pack_byte(int b, char pos):
  cdef unsigned int ans = 0
  ans = ((<unsigned int>b & 0x000000FF) << 8*pos)
  return ans

def pack_aps3_waveform_c(np.ndarray[np.complex128_t, ndim=1] wave):

  N = len(wave)
  assert N % 8 == 0

  cdef np.ndarray[np.complex128_t, ndim=1] wf_scaled = wave * ((1 << 15) - 1) #scale to size
  cdef np.ndarray[np.int32_t, ndim=1] wf_re = np.int32(np.around(np.real(wf_scaled)))
  cdef np.ndarray[np.int32_t, ndim=1] wf_im = np.int32(np.around(np.imag(wf_scaled)))

  cdef np.ndarray[np.uint32_t, ndim=1] packed_wf = np.empty(N, dtype=np.uint32)
  packed_wf.fill(0xBAAA_AAAD)

  cdef unsigned int ct
  cdef unsigned int ans
  cdef unsigned int x

  for ct in range(0, N-8, 8):

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_re[ct + 2*x] >> 8, x)
    packed_wf[ct] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_re[ct + 2*x], x)
    packed_wf[ct+1] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_re[ct+1 + 2*x] >> 8, x)
    packed_wf[ct+2] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_re[ct+1 + 2*x], x)
    packed_wf[ct+3] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_im[ct + 2*x] >> 8, x)
    packed_wf[ct+4] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_im[ct + 2*x], x)
    packed_wf[ct+5] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_im[ct+1 + 2*x] >> 8, x)
    packed_wf[ct+6] = ans

    ans = 0
    for x in range(4):
      ans = ans | pack_byte(wf_im[ct+1 + 2*x], x)
    packed_wf[ct+7] = ans

  return packed_wf
