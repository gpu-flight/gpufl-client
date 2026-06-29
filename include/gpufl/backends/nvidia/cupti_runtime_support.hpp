#pragma once

#include <cupti.h>

namespace gpufl {

class CuptiBackend;

bool IsInsufficientPrivilege(CUptiResult res);
void LogCuptiIfUnexpected(const char* scope, const char* op, CUptiResult res);

void PreloadMatchingPerfWorks();
bool WindowsInjectedProcess();
bool TryCurrentCudaContext(CUcontext* ctx);

void SetActiveCuptiBackend(CuptiBackend* backend);
CuptiBackend* GetActiveCuptiBackend();

}  // namespace gpufl
