# libchannelizer

Provides wrappers around Intel IPP filtering functions for use in the Auspex channelizer filter.


# Building

## IPP Deployable Shared Library

Intel provides a
[custom_library_tool](https://software.intel.com/en-us/articles/intel-integrated-performance-primitives-intel-ipp-for-windows-deploying-applications-with-intel-ipp-dlls)
to create a small shared library with only the needed functions for deployment
purposes. We can then link `libchannelizer` against this and not have the full
IPP as an Auspex dependency. We use the `-build` option to just build the shared
library but you can use the `-save` option to see what it is doing. It will put
the build products in an `intel64` folder and you'll have to copy them out.

### Linux

```shell
cryan $ cd /intel/compilers_and_libraries_2017.0.098/linux/ipp/tools/custom_library_tool
cryan $ ./ipp_custom_library_tool -build -n ippreduced -l "..../src/auspex/filters/libchannelizer/ipp_functions.txt" -o "..../src/auspex/filters/libchannelizer/" -c ~/intel/compilers_and_libraries
ippInit is added into list
ippGetLibVersion is added into list
ippsMalloc_8u is added into list
ippsFree is added into list
ippsFIRMRGetSize is added into list
ippsFIRMRInit_32f is added into list
ippsFIRMR_32f is added into list
ippsIIRGetStateSize_32f is added into list
ippsIIRInit_32f is added into list
ippsIIR_32f is added into list
ippsIIRSetDlyLine_32f is added into list
Library is successfully created!
```

### Windows

There is probably someway to get this to work the mingw64 but it's easier to
just install Visual Studio or Visual C++ Build Tools.

```shell
S C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\ipp\tools\custom_library_tool> .\ipp_custom_library_tool.exe -n libippreduced -l C:\Users\qlab\Documents\GitHub\Auspex\src
\auspex\filters\libchannelizer\ipp_functions.txt -o C:\Users\qlab\Documents\GitHub\Auspex\src\auspex\filters\libchannelizer\ -c 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries'
ippInit is added into list
ippGetLibVersion is added into list
ippsMalloc_8u is added into list
ippsFree is added into list
ippsFIRMRGetSize is added into list
ippsFIRMRInit_32f is added into list
ippsFIRMR_32f is added into list
ippsIIRGetStateSize_32f is added into list
ippsIIRInit_32f is added into list
ippsIIR_32f is added into list
ippsIIRSetDlyLine_32f is added into list
Library is successfully created!
PS C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.109\windows\ipp\tools\custom_library_tool>
```


## libchannelizer

We currently don't do any error checking so the compiler will warn about unused status variables.

### Linux

On Linux tell the linker to use a relative rpath so that when loading
`libchannelizer` into Python it can find `libippreduced`.

```shell
cryan $ source ~/intel/compilers_and_libraries/linux/ipp/bin/ippvars.sh intel64
cryan $ g++ -c -std=c++11 -fPIC -Wall -O3 channelizer.cpp
cryan $ g++ channelizer.o -shared -o libchannelizer.so -L. -lippreduced -Wl,-rpath=`$ORIGIN`
```

### Windows

We've used the usual msys2 mingw64 tool chain so all their shared libraries will
need to be on the path.

```shell
$ export IPPROOT="/c/Program Files x86)/IntelSWTools/compilers_and_libraries_2017.0.109/windows/ipp"
$ g++ -c -std=c++11 -fPIC -Wall -O3 channelizer.cpp
$ g++ channelizer.o -shared -o libchannelizer.dll -L. -lippreduced
```
