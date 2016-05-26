#pragma once

#include <zmq.hpp>
#include "timer.h"
#include "mymath.h"

#define KiB2B(_x)	((_x) * 1024)
#define MiB2B(_x)	(KiB2B(_x) * 1024)
#define GiB2B(_x)	(MiB2B(_x) * 1024)