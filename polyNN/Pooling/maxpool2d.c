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
#include "maxpool2d.h"


/* Array initialization. */
static
void init_array(int nn, int nd, int ih, int iw, int ow, int oh,
            DATA_TYPE POLYBENCH_4D(out_F,NN,ND,IH,IW,nn,nd,ih,iw),
            DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,OH,OW,nn,nd,oh,ow))
{
  int a, b, d, e;

  for (a = 0; a < nn; a++)
      for (b = 0; b < nd; b++)
          for (d = 0; d < iw; d++)
              for ( e = 0; e < ih; e++)
                  out_F[a][b][d][e] = (DATA_TYPE) 0;

  for (a = 0; a < nn; a++)
      for (b = 0; b < nd; b++)
          for (d = 0; d < ow; d++)
              for ( e = 0; e < oh; e++)
                  inp_F[a][b][d][e] = (DATA_TYPE) 0;
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
void maxpool2d_forward(int nn, int nd ,int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,            
            DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,iw,ih),
            DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,ow,oh))
{

  int n, d, r, c, row_st, row_nd, col_st, col_nd, val;
  #pragma scop

  for(n = 0; n < _PB_N; n++)
      for(d = 0; d < _PB_D; d++)
          for(r = 0; r < _PB_R; r++){
              row_st = r * sh;
              row_end = max(row_st + dh, ih);
              for(c = 0; c < _PB_C; c++){
                  col_st = c * sw;
                  col_nd = max(col_st + dw, iw);

                  val = -INF;
                  for(h = row_st; h < row_nd; h++)
                      for(w = col_st; w < col_nd; w++)
                          val = max(val, inp_F[n][d][h][w])

                  out_F[n][d][r][c] = val;
              }
          }

  #pragma endscop
}

static
void maxpool2d_backward(int nn, int nd ,int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,            
            DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,iw,ih),
            DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,ow,oh),
            DATA_TYPE POLYBENCH_4D(err_in,NN,ND,IH,IW,nn,nd,iw,ih),
            DATA_TYPE POLYBENCH_4D(err_out,NN,ND,OH,OW,nn,nd,ow,oh))
{

  int n, d, r, c, row_st, row_nd, col_st, col_nd;
  #pragma scop

  for(n = 0; n < _PB_N; n++)
      for(d = 0; d < _PB_D; d++)
          for(r = 0; r < _PB_R; r++){
              row_st = r * sh;
              row_end = max(row_st + dh, ih);
              for(c = 0; c < _PB_C; c++){
                  col_st = c * sw;
                  col_nd = max(col_st + dw, iw);

                  for(h = row_st; h < row_nd; h++)
                      for(w = col_st; w < col_nd; w++)
						  if(out_F[n][d][r][c] == inp_F[n][d][h][w])
							  err_in[n][d][h][w] += err_out[n][d][r][c];

              }
          }

  #pragma endscop
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. 
     inp - 4d Input matrix nn x nd x ih x iw
     (dh,dw) - pool size
     (sh,sw) - stride values
     out - 4d output matrix nn x nd x oh x ow
   */
  int nn = NN;
  int nd = ND;
  int ih = IH;
  int iw = IW;
  int dh = DH;
  int dw = DW;
  int sh = SH;
  int sw = SW;
  int oh = OH; // if sh == None -> oh = ih / sh  else oh = (ih - dh)/sh + 1
  int ow = OW; // if sw == None -> ow = iw / sw  else ow = (iw - dw)/sw + 1
    
  /* Variable declaration/allocation. */
  POLYBENCH_4D_ARRAY_DECL(inp_F,DATA_TYPE,NN,ND,IH,IW,nn,nd,ih,iw);
  POLYBENCH_4D_ARRAY_DECL(out_F,DATA_TYPE,NN,ND,OH,OW,nn,nd,oh,ow);
  POLYBENCH_4D_ARRAY_DECL(err_in,DATA_TYPE,NN,ND,IH,IW,nn,nd,ih,iw);
  POLYBENCH_4D_ARRAY_DECL(err_out,DATA_TYPE,NN,ND,OH,OW,nn,nd,oh,ow);

 
  /* Initialize array(s). */
  init_array (nn,nd,ih,iw,oh,ow,
          POLYBENCH_ARRAY(out_F),
          POLYBENCH_ARRAY(inp_F)
          POLYBENCH_ARRAY(err_in),
          POLYBENCH_ARRAY(err_out));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  maxpool2d_forward(nn, nd, ih, iw, oh ,ow, dh, dw, sh, sw,
          POLYBENCH_ARRAY(inp_F),
          POLYBENCH_ARRAY(out_F));

  maxpool2d_backward(nn, nd, ih, iw, oh ,ow, dh, dw, sh, sw,
          POLYBENCH_ARRAY(inp_F),
          POLYBENCH_ARRAY(out_F),
          POLYBENCH_ARRAY(err_in),
          POLYBENCH_ARRAY(err_out));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nn, nk, np, nq,  POLYBENCH_ARRAY(out_F)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(out_F);
  POLYBENCH_FREE_ARRAY(inp_F);

  return 0;
}
