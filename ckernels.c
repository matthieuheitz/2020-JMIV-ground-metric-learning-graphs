#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

#define sqr(x) x*x


#define EPSILON 1E-250


int myprint(const char *msg)
{
    printf("%s", msg);
    return 0;
}


void my_conv_batch(const double *u, double * result, double gamma, int W, int H, int D, int nvectors)
{
    int N = W*H*D;
    int M = max(max(W, H), D);
    double* kernel1d = (double*)malloc(M*sizeof(double));
    for (int i=0; i<M; i++) {
//        kernel1d[i] = max(EPSILON, exp(-i*i / gamma));
        double t = i/(double)(M-1);
        kernel1d[i] = max(EPSILON, exp(-t*t / gamma));
//        kernel1d[i] = max(EPSILON, exp(-sqr(i/(double)(M-1)) / gamma));
    }

    double* tmp = (double*)malloc(W*H*D*sizeof(double)); // allocating here ; otherwise not thread-safe

//    printf("Kernel 1d:\n");
//    for(int i=0;i<M;++i)
//    {
//        printf("%g, ",kernel1d[i]);
//    }

    for (int nv = 0; nv< nvectors; nv++) {

#pragma omp parallel for
        for (int d=0; d<D; d++) {
            for (int i=0; i<H; i++) {
                for (int j=0; j<W; j++) {
                    double conv = 0;
                    for (int k=0; k<W; k++) {
                        conv+=kernel1d[abs(j-k)]*u[nv*N + d*W*H + i*W + k];
                    }
                    // Stocké en transposé pour accélérer la lecture dans la prochaine boucle.
                    tmp[d*W*H + j*H+i] = conv;
                }
            }


            for (int j=0; j<W; j++) {
                for (int i=0; i<H; i++) {
                    double conv = 0;
                    for (int k=0; k<H; k++) {
                        conv+=kernel1d[abs(i-k)]*tmp[d*W*H + j*H + k];
                    }
                    result[nv*N + (i*W+j)*D + d] = conv;
                }
            }
        }

#pragma omp parallel for
        for (int i=0; i<H; i++) {
            for (int j=0; j<W; j++) {
                for (int d=0; d<D; d++) {
                    double conv = 0;
                    for (int k=0; k<D; k++) {
//                        conv+=kernel1d[abs(d-k)]*result[k + (i*W + j)*D];
                        conv+=kernel1d[abs(d-k)]*result[nv*N + k + (i*W + j)*D];
                    }
                    tmp[d*W*H + i*W+j] = conv;
                }
            }
        }
        memcpy(result+nv*N, tmp, W*H*D*sizeof(double));
    }
    free(tmp);
    free(kernel1d);
}


void convolution_batch_2d(const double* u, const double * kernel1d, double* result, int W, int H, int nvectors)
{
    int N = W*H;
    double* tmp = (double*)malloc(W*H*sizeof(double)); // allocating here ; otherwise not thread-safe

    for (int nv=0; nv<nvectors; nv++) {

#pragma omp parallel for
        for (int i=0; i<H; i++) {
            for (int j=0; j<W; j++) {
                double conv = 0;
                for (int k=0; k<W; k++) {
                    conv+=kernel1d[abs(j-k)]*u[nv*N + i*W + k];
                }
                tmp[i+j*H] = conv;
            }
        }

#pragma omp parallel for
        for (int j=0; j<W; j++) {
            for (int i=0; i<H; i++) {
                double conv = 0;
                for (int k=0; k<H; k++) {
                    conv+=kernel1d[abs(i-k)]*tmp[k + j*H];
                }
                result[nv*N + i*W+j] = conv;
            }
        }
    }
    free(tmp);

}


void convolution_batch(const double *u, const double * kernel1d, double * result, int W, int H, int D, int nvectors)
{
    int N = W*H*D;
    double* tmp = (double*)malloc(W*H*D*sizeof(double)); // allocating here ; otherwise not thread-safe

//    printf("\nKernel 1d:\n");
//    int M = max(max(W, H), D);
//    for(int i=0;i<M;++i)
//    {
//        printf("%g ",kernel1d[i]);
//    }
//    printf("\n");
//
//    printf("\nu:");
//    for(int i=0;i<N;++i)
//    {
//        if(i%10 == 0) printf("\n");
//        printf("%g ",u[i]);
//    }
//    printf("\n");
//
//    printf("\nresult:");
//    for(int i=0;i<N;++i)
//    {
//        if(i%10 == 0) printf("\n");
//        printf("%g ",result[i]);
//    }
//    printf("\n");

    for (int nv = 0; nv< nvectors; nv++) {

#pragma omp parallel for
        for (int d=0; d<D; d++) {
            for (int i=0; i<H; i++) {
                for (int j=0; j<W; j++) {
                    double conv = 0;
                    for (int k=0; k<W; k++) {
                        conv+=kernel1d[abs(j-k)]*u[nv*N + d*W*H + i*W + k];
                    }
                    // Stocké en transposé pour accélérer la lecture dans la prochaine boucle.
                    tmp[d*W*H + j*H+i] = conv;
                }
            }


            for (int j=0; j<W; j++) {
                for (int i=0; i<H; i++) {
                    double conv = 0;
                    for (int k=0; k<H; k++) {
                        conv+=kernel1d[abs(i-k)]*tmp[d*W*H + j*H + k];
                    }
                    result[nv*N + (i*W+j)*D + d] = conv;
                }
            }
        }

#pragma omp parallel for
        for (int i=0; i<H; i++) {
            for (int j=0; j<W; j++) {
                for (int d=0; d<D; d++) {
                    double conv = 0;
                    for (int k=0; k<D; k++) {
//                        conv+=kernel1d[abs(d-k)]*result[k + (i*W + j)*D];
                        conv+=kernel1d[abs(d-k)]*result[nv*N + k + (i*W + j)*D];
                    }
                    tmp[d*W*H + i*W+j] = conv;
                }
            }
        }
        memcpy(result+nv*N, tmp, W*H*D*sizeof(double));
    }
    free(tmp);

}

void convolution(const double * u, const double * kernel1d, double * result, int W, int H, int D)
{
    convolution_batch(u, kernel1d, result, W, H, D, 1);
}


void convolution_batch_build_kernel(const double * u, double * result, double gamma, int W, int H, int D, int nvectors)
{
    int M = max(max(W, H), D);
    double* kernel1d = (double*)malloc(M*sizeof(double));
    for (int i=0; i<M; i++) {
//        kernel1d[i] = max(EPSILON, exp(-i*i / gamma));
        double t = i/(double)(M-1);
        kernel1d[i] = max(EPSILON, exp(-t*t / gamma));
//        kernel1d[i] = max(EPSILON, exp(-sqr(i/(double)(M-1)) / gamma));
    }
    convolution_batch(u, kernel1d, result, W, H, D, nvectors);

    free(kernel1d);
}


void convolution_build_kernel(const double * u, double * result, double gamma, int W, int H, int D)
{
    convolution_batch_build_kernel(u, result, gamma, W, H, D, 1);
}



int main(int argc, char *argv[])
{
    // Test the convolution with 1 vector
    int n = 16;
    int N = n*n*n;
    int nv = 3;
    double gamma = 2*sqr(0.05);
    double * u = (double*)malloc(N*nv*sizeof(double));
    double * res = (double*)malloc(N*nv*sizeof(double));
    double * res2 = (double*)malloc(N*nv*sizeof(double));
    for(int i=0;i<N*nv;++i)
    {
        u[i] = 0.0;
        res[i] = 0.0;
    }
    u[0] = 1.0;
    if(nv>= 2) u[N + n-1] = 1.0;
    if(nv>= 3) u[2*N + n/2] = 1.0;
//    convolution(u,res,gamma,n,n,n);
    convolution_batch_build_kernel(u,res,gamma,n,n,n,nv);
    // Display only the first line of each vector
    printf("Vecteur:\n");
    for(int j=0;j<nv;++j)
    {
        for(int i=0;i<n;++i)
        {
            printf("%g, ",res[j*N + i]);
        }
        printf("\n\n");
    }

    int W = n;
    int H = n;
    int D = n;
    // Test convolution_batch
    int M = max(max(W, H), D);
    double* kernel1d = (double*)malloc(M*sizeof(double));
    for (int i=0; i<M; i++) {
//        kernel1d[i] = max(EPSILON, exp(-i*i / gamma));
        double t = i/(double)(M-1);
        kernel1d[i] = max(EPSILON, exp(-t*t / gamma));
//        kernel1d[i] = max(EPSILON, exp(-sqr(i/(double)(M-1)) / gamma));
    }
    convolution_batch(u,kernel1d,res2,n,n,n,nv);

    // Display only the first line of each vector
    printf("Vecteur:\n");
    for(int j=0;j<nv;++j)
    {
        for(int i=0;i<n;++i)
        {
            printf("%g, ",res2[j*N + i]);
        }
        printf("\n\n");
    }


//    // Display the whole vector
//    for(int i=0;i<N;++i)
//    {
//        if(i%n == 0 && i!=0) printf("\n");
//        if(i%(n*n) == 0 && i!=0) printf("\n");
//        printf("%g, ",res[i]);
//    }

    free(u);
    free(res);
    free(res2);
}