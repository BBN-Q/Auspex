#include "channelizer.h"

#include <iostream>

#include "ipp.h"
#include "ipps.h"

using std::cout;
using std::endl;

void init() {
  cout << "Initializing IPP Libraries!" << endl;
  const IppLibraryVersion *lib;
  ippInit();
  // print version info so we know something worked
  lib = ippGetLibVersion();
  cout << lib->Name << lib->Version << endl;
}

void filter_records_fir(float *coeffs, size_t num_taps, int decim_factor,
                        float *recs, size_t record_length, size_t num_records,
                        float *result) {
  IppStatus status;
  Ipp8u *filter_work = nullptr;
  IppsFIRSpec_32f *filter_spec = nullptr;
  int filter_spec_size, filter_work_size = 0;
  status = ippsFIRMRGetSize(num_taps, 1, decim_factor, ipp32f,
                            &filter_spec_size, &filter_work_size);
  // cout << "ippsFIRMRGetSize status = " << ippGetStatusString(status) << endl;
  // cout << "ippsFIRMRGetSize filter_spec_size = " << filter_spec_size << endl;
  // cout << "ippsFIRMRGetSize filter_work_size = " << filter_work_size << endl;
  filter_work = ippsMalloc_8u(filter_work_size);
  filter_spec = (IppsFIRSpec_32f *)ippsMalloc_8u(filter_spec_size);

  status =
      ippsFIRMRInit_32f(coeffs, num_taps, 1, 0, decim_factor, 0, filter_spec);
  // cout << "ippsFIRMRInit_32f status = " << ippGetStatusString(status) <<
  // endl;

  size_t output_length = record_length / decim_factor;

  for (size_t ct = 0; ct < num_records; ct++) {
    status = ippsFIRMR_32f(recs + ct * record_length,
                           result + ct * output_length, output_length,
                           filter_spec, nullptr, nullptr, filter_work);
    // cout << "ippsFIRMR_32f status = " << ippGetStatusString(status) << endl;
  }

  ippsFree(filter_work);
  ippsFree(filter_spec);
}

void filter_records_iir(float *coeffs, size_t order, float *recs,
                        size_t record_length, size_t num_records, float *filtered) {
  IppStatus status;
  IppsIIRState_32f *filter_state = nullptr;
  Ipp8u *filter_work = nullptr;
  int filter_work_size = 0;

  status = ippsIIRGetStateSize_32f(order, &filter_work_size);
  // cout << "ippsIIRGetStateSize_32f status = " << ippGetStatusString(status)
  //      << endl;
  // cout << "ippsIIRGetStateSize_32f filter_work_size = " << filter_work_size
  //      << endl;

  filter_work = ippsMalloc_8u(filter_work_size);

  status = ippsIIRInit_32f(&filter_state, coeffs, order, nullptr, filter_work);

  for (size_t ct = 0; ct < num_records; ct++) {
    status =
        ippsIIR_32f(recs + ct * record_length, filtered + ct* record_length, record_length, filter_state);
    // cout << "ippsIIR_32f_I status = " << ippGetStatusString(status) << endl;
    status = ippsIIRSetDlyLine_32f(filter_state, nullptr);
    // cout << "ippsIIRSetDlyLine_32f status = " << ippGetStatusString(status)
    //  << endl;
  }

  ippsFree(filter_work);
}
