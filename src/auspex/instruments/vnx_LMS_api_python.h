// Include file for 64 Bit Vaunix Lab Brick LMS Synthesizer DLL
//
// 10/2013	RD	64 Bit DLL version.
//

void fnLMS_SetTestMode(bool testmode);
int fnLMS_GetNumDevices();
int fnLMS_GetDevInfo(unsigned int *ActiveDevices);
int fnLMS_GetModelNameA(unsigned int deviceID, char *ModelName);
// int fnLMS_GetModelNameW(unsigned int deviceID, wchar_t *ModelName);
int fnLMS_InitDevice(unsigned int deviceID);
int fnLMS_CloseDevice(unsigned int deviceID);
int fnLMS_GetSerialNumber(unsigned int deviceID);
int fnLMS_GetDLLVersion();
int fnLMS_SetFrequency(unsigned int deviceID, int frequency);
int fnLMS_SetStartFrequency(unsigned int deviceID, int startfrequency);
int fnLMS_SetEndFrequency(unsigned int deviceID, int endfrequency);
int fnLMS_SetSweepTime(unsigned int deviceID, int sweeptime);
int fnLMS_SetPowerLevel(unsigned int deviceID, int powerlevel);
int fnLMS_SetRFOn(unsigned int deviceID, bool on);
int fnLMS_SetPulseOnTime(unsigned int deviceID, float pulseontime);
int fnLMS_SetPulseOffTime(unsigned int deviceID, float pulseofftime);
int fnLMS_EnableInternalPulseMod(unsigned int deviceID, bool on);
int fnLMS_SetUseExternalPulseMod(unsigned int deviceID, bool external);
int fnLMS_SetFastPulsedOutput(unsigned int deviceID, float pulseontime, float pulsereptime, bool on);
int fnLMS_SetUseInternalRef(unsigned int deviceID, bool internal);
int fnLMS_SetSweepDirection(unsigned int deviceID, bool up);
int fnLMS_SetSweepMode(unsigned int deviceID, bool mode);
int fnLMS_SetSweepType(unsigned int deviceID, bool swptype);
int fnLMS_StartSweep(unsigned int deviceID, bool go);
int fnLMS_SaveSettings(unsigned int deviceID);
int fnLMS_GetFrequency(unsigned int deviceID);
int fnLMS_GetStartFrequency(unsigned int deviceID);
int fnLMS_GetEndFrequency(unsigned int deviceID);
int fnLMS_GetSweepTime(unsigned int deviceID);
int fnLMS_GetRF_On(unsigned int deviceID);
int fnLMS_GetUseInternalRef(unsigned int deviceID);
int fnLMS_GetPowerLevel(unsigned int deviceID);
int fnLMS_GetAbsPowerLevel(unsigned int deviceID);
int fnLMS_GetMaxPwr(unsigned int deviceID);
int fnLMS_GetMinPwr(unsigned int deviceID);
int fnLMS_GetMaxFreq(unsigned int deviceID);
int fnLMS_GetMinFreq(unsigned int deviceID);
float fnLMS_GetPulseOnTime(unsigned int deviceID);
float fnLMS_GetPulseOffTime(unsigned int deviceID);
int fnLMS_GetPulseMode(unsigned int deviceID);
int fnLMS_GetHasFastPulseMode(unsigned int deviceID);
int fnLMS_GetUseInternalPulseMod(unsigned int deviceID);
int fnLMS_GetDeviceStatus(unsigned int deviceID);
