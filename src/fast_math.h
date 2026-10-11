#pragma once


namespace genetic {
namespace detail {

#if defined(__FMA__)
#define GENETIC_FAST_MATH 1
void sinf_block(const float *in, float *out, int n);
void cosf_block(const float *in, float *out, int n);
void logf_block(const float *in, float *out, int n);
void expf_block(const float *in, float *out, int n);
void log1pf_block(const float *in, float *out, int n);
#endif

bool use_fast_math_blocks();

}
}
