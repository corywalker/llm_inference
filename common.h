#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <thread>

extern bool verbose_g;

#define LOG_VERBOSE(x)             \
  do {                             \
    if (verbose_g) {               \
      std::cout << x << std::endl; \
    }                              \
  } while (0)
#define VERBOSE(x)   \
  do {               \
    if (verbose_g) { \
      x;             \
    }                \
  } while (0)

#endif  // COMMON_H
