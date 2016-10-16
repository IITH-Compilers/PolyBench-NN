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
 *  FIXME : Initialize arrays properly, 
 *      set appropriate data sizes in header file, 
 *      infer P and Q (output dimensions from others) 
 *      
 *
 *      Implemented by referring cuDNN paper.
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
#include "rnn.h"


/* Array initialization. */
static
void init_array_forward(int nt, int np, int ns, int nq,
            DATA_TYPE POLYBENCH_2D(out_F,NT,NQ,nt,nq),
            DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
            DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
            DATA_TYPE POLYBENCH_2D(U,NS,NP,ns,np),
            DATA_TYPE POLYBENCH_2D(W,NS,NS,ns,ns),
            DATA_TYPE POLYBENCH_2D(V,NQ,NS,nq,ns))
{
  int a, b;

  for (a = 0; a < nt; a++)
    for (b = 0; b < nq; b++) 
        out_F[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < nt; a++)
    for (b = 0; b < ns; b++) 
        s_F[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < nt; a++)
    for (b = 0; b < np; b++) 
        inp_F[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < ns; a++)
    for (b = 0; b < np; b++) 
        U[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < ns; a++)
    for (b = 0; b < ns; b++) 
        W[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < nq; a++)
    for (b = 0; b < ns; b++) 
        V[a][b] = (DATA_TYPE) 0;
}


static
void init_array_backward(int nt, int np, int ns, int nq,
            DATA_TYPE POLYBENCH_2D(err_out,NT,NQ,nt,nq),
            DATA_TYPE POLYBENCH_2D(del_U,NS,NP,ns,np),
            DATA_TYPE POLYBENCH_2D(del_W,NS,NS,ns,ns),
            DATA_TYPE POLYBENCH_2D(del_V,NQ,NS,nq,ns))
{
  int a, b;

  for (a = 0; a < nt; a++)
    for (b = 0; b < nq; b++) 
        err_out[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < ns; a++)
    for (b = 0; b < np; b++) 
        del_U[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < ns; a++)
    for (b = 0; b < ns; b++) 
        del_W[a][b] = (DATA_TYPE) 0;

  for (a = 0; a < nq; a++)
    for (b = 0; b < ns; b++) 
        del_V[a][b] = (DATA_TYPE) 0;
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


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void rnn_forward(int nt, int np, int ns, int nq,            
            DATA_TYPE POLYBENCH_2D(out_F,NT,NQ,nt,nq),
            DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
            DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
            DATA_TYPE POLYBENCH_2D(U,NS,NP,ns,np),
            DATA_TYPE POLYBENCH_2D(W,NS,NS,ns,ns),
            DATA_TYPE POLYBENCH_2D(V,NQ,NS,nq,ns))
{
  int t, p, q, s1, s2;
  #pragma scop

  for (t = 0; t < _PB_NT; t++)
  {    
      for(s1 = 0; s1 < _PB_NS; s1++)
          for(p = 0; p < _PB_NP; p++)
              s_F[t][s1] += U[s1][p] * inp_F[t][p];

          for(s2 = 0; s2 < _PB_NS; s2++)
              s_F[t][s1] += W[s1][s2] * s_F[(t-1+NT) % NT][s2];
      
      for(q = 0; q < _PB_NQ; q++){
          for(s1 = 0; s1 < _PB_NS; s1++){
              out_F[t][q] += V[q][s1] * s_F[t][s1];
  }
  #pragma endscop
}


static
void rnn_backward(int nt, int np, int ns, int nq, int ndel, int bptt_trunc,  
            DATA_TYPE POLYBENCH_2D(inp_F,NT,NQ,nt,nq),
            DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
            DATA_TYPE POLYBENCH_2D(W,NS,NS,ns,ns),
            DATA_TYPE POLYBENCH_2D(V,NQ,NS,nq,ns),
            DATA_TYPE POLYBENCH_2D(err_out,NT,NQ,nt,nq),
            DATA_TYPE POLYBENCH_2D(del_T,NDEL,NS,ndel,ns),
            DATA_TYPE POLYBENCH_2D(del_U,NS,NP,ns,np),
            DATA_TYPE POLYBENCH_2D(del_W,NS,NS,ns,ns),
            DATA_TYPE POLYBENCH_2D(del_V,NQ,NS,nq,ns))
{
  int t, p, q, s, flag;
  #pragma scop

  /*
     Consider del_T to be array of array of size 2 x ns
  */

  for (t = _PB_NT - 1; t > 0; t--)
  {  
      flag = 0;
      for(q = 0; q < _PB_NQ; q++)
          for(s = 0; s < _PB_NS; s++)
              del_V[q][s] = err_out[t][q] * s_F[t][s];
      
      // Loop swap may help
      for(s = 0; s < _PB_NS; s++)
      {
          del_T[flag][s] = (DATA_TYPE) 0;
          for(q = 0; q < _PB_NQ; q++)
              del_T[flag][s] += V[q][s] * err_out[t][q];
      }

      for(step = t + 1; step > max(0, t - bptt_trunc); step--)
      {
          for(r = 0; r < _PB_NS; r++)
              for(s = 0; s < _PB_NS; s++)
                  del_W[r][s] = del_T[flag][r] * s_F[step - 1][s];	// step - 1 need to check

          for(s = 0; s < _PB_NS; s++)
              for(p = 0; p < _PB_NP; p++)
                  del_U[s][p] += del_T[flag][s] * inp_F[step][p];

          for(r = 0; r < _PB_NS; r++)
          {
              del_T[1 - flag][r] = (DATA_TYPE) 0;
              for(s = 0; s < _PB_NS; s++)
                  del_T[1 - flag][r] += del_T[flag][s] * W[s][r];   // TODO: Verify

          }

          flag = 1 - flag;
      }
  }
  #pragma endscop

}
int main(int argc, char** argv)
{
  /* Retrieve problem size. 
     x - Input sequence         nt x np
     s - State at each step     nt x ns
     o - Output at each step    nt x nq
     U - Matrix multiplied with x       ns x np 
     W - Matrix multiplied with s(t-1)  ns x ns
     V - Matrix multiplied with s(t)    nq x ns

     Please refer to this link to understand the connections
     http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/rnn.jpg
   */
  int nt = NT;
  int np = NP;
  int nq = NQ;
  int ns = NS;
  int ndel = NDEL;  // 2
  int bptt_trunc = BT;
    
  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(out_F,DATA_TYPE,NT,NQ,nt,nq);
  POLYBENCH_2D_ARRAY_DECL(s_F,DATA_TYPE,NT,NS,nt,ns);
  POLYBENCH_2D_ARRAY_DECL(inp_F,DATA_TYPE,NT,NP,nt,np);
  POLYBENCH_2D_ARRAY_DECL(U,DATA_TYPE,NS,NP,ns,np);
  POLYBENCH_2D_ARRAY_DECL(W,DATA_TYPE,NS,NS,ns,ns);
  POLYBENCH_2D_ARRAY_DECL(V,DATA_TYPE,NQ,NS,nq,ns);

  /* Required for backprop */
  POLYBENCH_2D_ARRAY_DECL(err_out,DATA_TYPE,NT,NQ,nt,nq);
  POLYBENCH_2D_ARRAY_DECL(del_T,DATA_TYPE,NDEL,NS,ndel,ns);
  POLYBENCH_2D_ARRAY_DECL(del_U,DATA_TYPE,NS,NP,ns,np);
  POLYBENCH_2D_ARRAY_DECL(del_W,DATA_TYPE,NS,NS,ns,ns);
  POLYBENCH_2D_ARRAY_DECL(del_V,DATA_TYPE,NQ,NS,nq,ns);


  /* Initialize array(s). */
  init_array_forward (nt,np,ns,nq,
          POLYBENCH_ARRAY(out_F),
          POLYBENCH_ARRAY(s_F),
          POLYBENCH_ARRAY(inp_F),
          POLYBENCH_ARRAY(U),
          POLYBENCH_ARRAY(W),
          POLYBENCH_ARRAY(V));

  init_array_backward (nt,np,ns,nq,
          POLYBENCH_ARRAY(err_out),
          POLYBENCH_ARRAY(del_T),
          POLYBENCH_ARRAY(del_U),
          POLYBENCH_ARRAY(del_W),
          POLYBENCH_ARRAY(del_V));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  rnn_forward(nt, np, ns, nq,
          POLYBENCH_ARRAY(out_F),
          POLYBENCH_ARRAY(s_F),
          POLYBENCH_ARRAY(inp_F),
          POLYBENCH_ARRAY(U),
          POLYBENCH_ARRAY(W),
          POLYBENCH_ARRAY(V));

  rnn_backward(nt, np, ns, nq, ndel, bptt_trunc,  
            POLYBENCH_ARRAY(inp_F),
            POLYBENCH_ARRAY(s_F),
            POLYBENCH_ARRAY(W),
            POLYBENCH_ARRAY(V),
            POLYBENCH_ARRAY(err_out),
            POLYBENCH_ARRAY(del_T),
            POLYBENCH_ARRAY(del_U),
            POLYBENCH_ARRAY(del_W),
            POLYBENCH_ARRAY(del_V));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nn, nk, np, nq,  POLYBENCH_ARRAY(out_F)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(out_F);
  POLYBENCH_FREE_ARRAY(s_F);
  POLYBENCH_FREE_ARRAY(inp_F);
  POLYBENCH_FREE_ARRAY(U);
  POLYBENCH_FREE_ARRAY(W);
  POLYBENCH_FREE_ARRAY(V);
  POLYBENCH_FREE_ARRAY(del_T);    
  POLYBENCH_FREE_ARRAY(del_U);
  POLYBENCH_FREE_ARRAY(del_W);
  POLYBENCH_FREE_ARRAY(del_V);

  return 0;
}
