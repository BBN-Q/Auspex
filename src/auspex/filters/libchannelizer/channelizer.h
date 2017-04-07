#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void init();
void filter_records_fir(float *coeffs, size_t num_taps, int decim_factor,
                        float *recs, size_t record_length, size_t num_records,
                        float *result);

void filter_records_iir(float *coeffs, size_t order, float *recs,
                   size_t record_length, size_t num_records, float *filtered);

#ifdef __cplusplus
}
#endif
