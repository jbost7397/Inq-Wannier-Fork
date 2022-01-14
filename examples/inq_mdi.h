#ifndef INQ_MDI
#define INQ_MDI

// ensure that symbols are exported to Windows .dll files
#ifdef _WIN32
  #define DllExport   __declspec( dllexport )
#else
  #define DllExport
#endif

extern "C" DllExport int MDI_Plugin_init_inqmdi();

#endif
