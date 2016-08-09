#pragma once

#include <zmq.hpp>
#include "timer.h"
#include "mymath.h"

#define KiB2B(_x)	((_x) * 1024ul)
#define MiB2B(_x)	(KiB2B((_x)) * 1024ul)
#define GiB2B(_x)	(MiB2B((_x)) * 1024ul)

#define B2KiB(_x)	((_x) / 1024.0)
#define B2MiB(_x)	(B2KiB((_x)) / 1024.0)
#define B2GiB(_x)	(B2MiB((_x)) / 1024.0)
