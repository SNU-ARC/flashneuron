#ifdef _WIN32
	#include <windows.h>
#else
	#include <pthread.h>
#endif

#include <cpuinfo.h>
#include <cpuinfo/internal-api.h>
#include <cpuinfo/log.h>

#ifdef __APPLE__
	#include "TargetConditionals.h"
#endif


#ifdef _WIN32
	static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;
#else
	static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

bool CPUINFO_ABI cpuinfo_initialize(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	#if defined(__MACH__) && defined(__APPLE__)
		pthread_once(&init_guard, &cpuinfo_x86_mach_init);
	#elif defined(__linux__)
		pthread_once(&init_guard, &cpuinfo_x86_linux_init);
	#elif defined(_WIN32)
		InitOnceExecuteOnce(&init_guard, &cpuinfo_x86_windows_init, NULL, NULL);
	#else
		cpuinfo_log_error("operating system is not supported in cpuinfo");
	#endif
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	#if defined(__linux__)
		pthread_once(&init_guard, &cpuinfo_arm_linux_init);
	#elif defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
		pthread_once(&init_guard, &cpuinfo_arm_mach_init);
	#else
		cpuinfo_log_error("operating system is not supported in cpuinfo");
	#endif
#else
	cpuinfo_log_error("processor architecture is not supported in cpuinfo");
#endif
	return cpuinfo_is_initialized;
}

void CPUINFO_ABI cpuinfo_deinitialize(void) {
}
