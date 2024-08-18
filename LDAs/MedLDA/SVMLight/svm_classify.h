
#include "svm_common.h"

int classify (int argc, char* argv[]);
int classify(MODEL* model, DOC** docs, double *target, int totdoc);