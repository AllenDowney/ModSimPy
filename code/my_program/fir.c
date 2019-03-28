#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define max(a, b) (a>b)?a:b

float h[]=       // Impulse response of FIR filter.
{
		-2,    10,    14,     7,    -7,   -17,   -13 ,    3,
		19,    21,     4,   -21,   -32,   -16,    18 ,   43,
		34,    -8,   -51,   -56,   -11,    53,    81 ,   41,
		-44,  -104,   -81,    19,   119,   129,    24 , -119,
		-178,   -88,    95,   222,   171,   -41,  -248 , -266,
		-50,   244,   366,   181,  -195,  -457,  -353 ,   85,
		522,   568,   109,  -540,  -831,  -424,   474 , 1163,
		953,  -245, -1661, -2042,  -463,  2940,  6859 , 9469,
		9469,  6859,  2940,  -463, -2042, -1661,  -245 ,  953,
		1163,   474,  -424,  -831,  -540,   109,   568 ,  522,
		85,  -353,  -457,  -195,   181,   366,   244 ,  -50,
		-266,  -248,   -41,   171,   222,    95,   -88 , -178,
		-119,    24,   129,   119,    19,   -81,  -104 ,  -44,
		41,    81,    53,   -11,   -56,   -51,    -8 ,   34,
		43,    18,   -16,   -32,   -21,     4,    21 ,   19,
		3,   -13,   -17,    -7,     7,    14,    10 ,   -2
};


float* fir(float* input, int n_sample)
{
  float* output = (float*)malloc(sizeof(float)*n_sample);
  for(int i=0; i<n_sample; i++)
  {
    float acc = 0;
    for(int j = 0; j < 128; j++)
      acc += input[max(0, i-j)]*h[j];
    output[i] = acc;
  }
  return output;
}

int main() {
  float fs = 80000;
  float signal_len = 0.04;
  float n_sample = fs*signal_len;
  float* t = (float*)malloc(sizeof(float)*n_sample);


  float freq = 1000;
  float freq_noise = 15000;
  float* sin_wave = (float*)malloc(sizeof(float)*n_sample);
  float* noise_wave = (float*)malloc(sizeof(float)*n_sample);
  float* mixed_wave = (float*)malloc(sizeof(float)*n_sample);


  for(int i=0; i<n_sample; i++)
  {
    t[i] = (float)i/fs;
    sin_wave[i] = sin(2*M_PI*freq*t[i]);
    noise_wave[i] = 0.5*sin(2*M_PI*freq_noise*t[i]);
    mixed_wave[i] = sin_wave[i] + noise_wave[i];
  }

  float* output_signal = fir(mixed_wave, n_sample);

  for (int i = 1000;i<1500;i++)
  {
    printf("%f ", output_signal[i]);
  }

	return 0;
}
