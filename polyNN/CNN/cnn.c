/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 *
 *
 *
 *	FIXME : Initialize arrays properly, 
 *		set appropriate data sizes in header file, 
 *		infer P and Q (output dimensions from others) 
 *		
 *
 *		Implemented by referring cuDNN paper.
 *
 *
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "cnn.h"


/* Array initialization. */
static
void init_array(int nn, int nk,int np,int nq,int nc,int nr,int ns,int nw,int nh, 
			DATA_TYPE POLYBENCH_4D(out_F,NN,NK,NP,NQ,nn,nk,np,nq),
			DATA_TYPE POLYBENCH_4D(W,NK,NC,NR,NS,nk,nc,nr,ns),
			DATA_TYPE POLYBENCH_4D(inp_F,NN,NC,NH,NW,nn,nc,nh,nw))
{
  int a, b, e, d;

  for (a = 0; a < nn; a++)
    for (b = 0; b < nk; b++) 
	for (e = 0; e < np; e++) 
	  for (d = 0; d < nq; d++) 
		out_F[a][b][e][d] = (DATA_TYPE) 0;

   for (a = 0; a < nk; a++)
    for (b = 0; b < nc; b++) 
	for (e = 0; e < nr; e++) 
	  for (d = 0; d < ns; d++)
		 W[a][b][e][d] = (DATA_TYPE) 0;

   for (a = 0; a < nn; a++)
    for (b = 0; b < nc; b++) 
	for (e = 0; e < nh; e++) 
	  for (d = 0; d < nw; d++)
		 W[a][b][e][d] = (DATA_TYPE) 0;	
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nn, int nk, int np, int nq, DATA_TYPE POLYBENCH_4D(out_F,NN,NK,NP,NQ,nn,nk,np,nq))
{
  int a, b, e, d;

  for (a = 0; a < nn; a++)
    for (b = 0; b < nk; b++) 
	for (e = 0; e < np; e++) 
	  for (d = 0; d < nq; d++) 
    {
	fprintf (stderr, DATA_PRINTF_MODIFIER, out_F[a][b][e][d]);
	if ((a*nn*nk*np + b * nn * nk + e *np + d) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}

inline int get_index(int p, int u, int R, int r)
{
	return p*u+R-r-1;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void cnn_forward(int nn, int nk,int np,int nq,int nc,int nr,int ns,int nw,int nh,int u,int v,  
			DATA_TYPE POLYBENCH_4D(out_F,NN,NK,NP,NQ,nn,nk,np,nq),
			DATA_TYPE POLYBENCH_4D(W,NK,NC,NR,NS,nk,nc,nr,ns),
			DATA_TYPE POLYBENCH_4D(inp_F,NN,NC,NH,NW,nn,nc,nh,nw))
{
  int n, k, p, q, c, r, s;
  #pragma scop
  for (n = 0; n < _PB_N; n++)
    for (k = 0; k < _PB_K; k++)
       for (p = 0; p < _PB_P; p++)
     	    for (q = 0; q < _PB_Q; q++)
		for (c = 0; c < _PB_C; c++)
			for (r = 0; r < _PB_R; r++)
				for (s = 0; s < _PB_S; s++)	
					 out_F[n][k][p][q] += W[k][c][r][s] * inp_F[n][c][get_index(p,u,NR,r)][get_index(q,v,NS,s)];
  #pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. 
   nn -> Batch size
   nk -> Number of output feature maps
   np -> Output matrix height
   nq -> Output matrix width
   nc -> Number of input feature maps
   nr -> Filter height
   ns -> Filter width
   nh -> Input matrix height
   nw -> Input matrix width
   */
  int nn = NN;	
  int nk = NK;
  int np = NP;
  int nq = NQ;
  int nc = NC;
  int nr = NR;
  int ns = NS;
  int nw = NW;
  int nh = NH;
  int nu = NU;
  int nv = NV;	
	
  /* Variable declaration/allocation. */
  POLYBENCH_4D_ARRAY_DECL(out_F,DATA_TYPE,NN,NK,NP,NQ,nn,nk,np,nq);
  POLYBENCH_4D_ARRAY_DECL(W,DATA_TYPE,NK,NC,NR,NS,nk,nc,nr,ns);
  POLYBENCH_4D_ARRAY_DECL(inp_F,DATA_TYPE,NN,NC,NH,NW,nn,nc,nh,nw);
 

  /* Initialize array(s). */
  init_array (nn,nk,np,nq,nc,nr,ns,nw,nh,
	      POLYBENCH_ARRAY(out_F),
	      POLYBENCH_ARRAY(W),
	      POLYBENCH_ARRAY(inp_F));


  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  cnn_forward(nn, nk, np, nq, nc, nr, ns, nw, nh,nu,nv,
	      POLYBENCH_ARRAY(out_F),
	      POLYBENCH_ARRAY(W),
	      POLYBENCH_ARRAY(inp_F));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nn, nk, np, nq,  POLYBENCH_ARRAY(out_F)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(out_F);
  POLYBENCH_FREE_ARRAY(W);
  POLYBENCH_FREE_ARRAY(inp_F);

  return 0;
}
