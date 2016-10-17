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
 *      
 *
 *	FIXME : Update reference link.
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
#include "maxpool.h"

#include <limits.h>

/* Array initialization. */
	static
void init_array(int nn, int nd, int ih, int iw, int oh, int ow,
		DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,oh,ow),
		DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(err_in,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(err_out,NN,ND,OH,OW,nn,nd,oh,ow))
{
	int a, b, d, e;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++)
			for (d = 0; d < oh; d++)
				for ( e = 0; e < ow; e++)
				{
					out_F[a][b][d][e] = (DATA_TYPE) (a*b + d*e % nn);
					err_out[a][b][d][e] = (DATA_TYPE) (a+b+d+e % nn);		
				}

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++)
			for (d = 0; d < iw; d++)
				for ( e = 0; e < ih; e++)
				{
					inp_F[a][b][d][e] = (DATA_TYPE) (a*b + d*e % nd);
					err_in[a][b][d][e] = (DATA_TYPE) (a+b+ d+e % nd);
				}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
	static
void print_array_fwd(int nn, int nd, int oh, int ow, DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,oh,ow))
{
	int a, b, e, d;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++) 
			for (e = 0; e < oh; e++) 
				for (d = 0; d < ow; d++) 
				{
					fprintf (stderr, DATA_PRINTF_MODIFIER, out_F[a][b][e][d]);
					if ((a*nd*oh*ow + b*oh*ow + e*ow + d) % 20 == 0) fprintf (stderr, "\n");
				}
	fprintf (stderr, "\n");
}

	static
void print_array_bwd(int nn, int nd, int ih, int iw, DATA_TYPE POLYBENCH_4D(err_in,NN,ND,IH,IW,nn,nd,ih,iw))
{
	int a, b, e, d;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nd; b++) 
			for (e = 0; e < ih; e++) 
				for (d = 0; d < iw; d++) 
				{
					fprintf (stderr, DATA_PRINTF_MODIFIER, err_in[a][b][e][d]);
					if ((a*nd*ih*iw + b * ih * iw + e *iw + d) % 20 == 0) fprintf (stderr, "\n");
				}
	fprintf (stderr, "\n");
}



/* Main computational kernel. The whole function will be timed,

   including the call and return. */
	static
void maxpool2d_forward(int nn, int nd ,int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,            
		DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,oh,ow))
{

	int n, d, r, c, h, w, row_st, row_end, col_st, col_nd, val;
#pragma scop

	for(n = 0; n < _PB_NN; n++)
		for(d = 0; d < _PB_ND; d++)
			for(r = 0; r < _PB_NR; r++){
				// FIXME Non affine	
				row_st = r * sh;
				row_end = MAX(row_st + dh, ih);
				for(c = 0; c < _PB_NC; c++){
					// FIXME Non affine
					col_st = c * sw;
					col_nd = MAX(col_st + dw, iw);

					val = INT_MIN;
					for(h = row_st; h < row_end; h++)
						for(w = col_st; w < col_nd; w++)
							val = MAX(val, inp_F[n][d][h][w]);
					out_F[n][d][r][c] = val;
				}
			}

#pragma endscop
}

	static
void maxpool2d_backward(int nn, int nd ,int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,            
		DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,oh,ow),
		DATA_TYPE POLYBENCH_4D(err_in,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(err_out,NN,ND,OH,OW,nn,nd,oh,ow))
{

	int n, d, r, c, h, w, row_st, row_end, col_st, col_nd;
#pragma scop

	for(n = 0; n < _PB_NN; n++)
		for(d = 0; d < _PB_ND; d++)
			for(r = 0; r < _PB_NR; r++){
				row_st = r * sh;
				row_end = MAX(row_st + dh, ih);
				for(c = 0; c < _PB_NC; c++){
					col_st = c * sw;
					col_nd = MAX(col_st + dw, iw);

					for(h = row_st; h < row_end; h++)
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
			POLYBENCH_ARRAY(inp_F),
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
	polybench_prevent_dce(print_array_fwd(nn,nd,ow,oh,POLYBENCH_ARRAY(out_F)));
	polybench_prevent_dce(print_array_bwd(nn,nd,iw,ih,POLYBENCH_ARRAY(err_in)));

	/* Be clean. */
	POLYBENCH_FREE_ARRAY(out_F);
	POLYBENCH_FREE_ARRAY(inp_F);
	POLYBENCH_FREE_ARRAY(err_in);
	POLYBENCH_FREE_ARRAY(err_out);

	return 0;
}
