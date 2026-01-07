#ifndef FFN_H
#define FFN_H

void geglu(const float* x, float* out, int B);
void initialize_once();

extern float *Wu;
extern float* Wv;
extern float* Wo;

#endif