#include <omp.h>
#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

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
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#define check_equal(x, y) (((x) == (y)) ? (1) : (0))

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "avgpool.h"

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
void sumpool2d_forward(int nn, int nd ,int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,            
		DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,oh,ow))
{

	int n, d, r, c, h, w, row_st, row_end, col_st, col_nd;
	DATA_TYPE val;
/* Copyright (C) 1991-2015 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses ISO/IEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0.  */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6, t7;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NC >= 1) && (_PB_ND >= 1) && (_PB_NN >= 1) && (_PB_NR >= 1)) {
  if ((_PB_NC >= ceild(iw+10,10)) && (_PB_NR >= ceild(ih+10,10)) && (dh >= 1) && (dw >= 1) && (ih >= 1) && (iw >= 1)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=floord(ih-1,10);t3++) {
          for (t4=0;t4<=floord(iw-1,10);t4++) {
            val = 0;;
            for (t6=10*t3;t6<=min(ih-1,10*t3+dh-1);t6++) {
              for (t7=10*t4;t7<=min(iw-1,10*t4+dw-1);t7++) {
                val += inp_F[t1][t2][t6][t7];;
              }
            }
            out_F[t1][t2][t3][t4] = val;;
          }
          for (t4=ceild(iw,10);t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
        for (t3=ceild(ih,10);t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if ((_PB_NC >= ceild(iw+10,10)) && (_PB_NR <= floord(ih+9,10)) && (dh >= 1) && (dw >= 1) && (iw >= 1)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=floord(iw-1,10);t4++) {
            val = 0;;
            for (t6=10*t3;t6<=min(ih-1,10*t3+dh-1);t6++) {
              for (t7=10*t4;t7<=min(iw-1,10*t4+dw-1);t7++) {
                val += inp_F[t1][t2][t6][t7];;
              }
            }
            out_F[t1][t2][t3][t4] = val;;
          }
          for (t4=ceild(iw,10);t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if ((_PB_NC <= floord(iw+9,10)) && (_PB_NR >= ceild(ih+10,10)) && (dh >= 1) && (dw >= 1) && (ih >= 1)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=floord(ih-1,10);t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            for (t6=10*t3;t6<=min(ih-1,10*t3+dh-1);t6++) {
              for (t7=10*t4;t7<=min(iw-1,10*t4+dw-1);t7++) {
                val += inp_F[t1][t2][t6][t7];;
              }
            }
            out_F[t1][t2][t3][t4] = val;;
          }
        }
        for (t3=ceild(ih,10);t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if ((_PB_NC <= floord(iw+9,10)) && (_PB_NR <= floord(ih+9,10)) && (dh >= 1) && (dw >= 1)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            for (t6=10*t3;t6<=min(ih-1,10*t3+dh-1);t6++) {
              for (t7=10*t4;t7<=min(iw-1,10*t4+dw-1);t7++) {
                val += inp_F[t1][t2][t6][t7];;
              }
            }
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if ((dh >= 1) && (dw >= 1) && (ih >= 1) && (iw <= 0)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if ((dh >= 1) && (dw <= 0) && (ih >= 1)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if ((dh >= 1) && (ih <= 0)) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
  if (dh <= 0) {
    for (t1=0;t1<=_PB_NN-1;t1++) {
      for (t2=0;t2<=_PB_ND-1;t2++) {
        for (t3=0;t3<=_PB_NR-1;t3++) {
          for (t4=0;t4<=_PB_NC-1;t4++) {
            val = 0;;
            out_F[t1][t2][t3][t4] = val;;
          }
        }
      }
    }
  }
}
/* End of CLooG code */
}

	static
void sumpool2d_backward(int nn, int nd ,int ih, int iw, int ow, int oh, int dh, int dw, int sh, int sw,            
		DATA_TYPE POLYBENCH_4D(inp_F,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(out_F,NN,ND,OH,OW,nn,nd,oh,ow),
		DATA_TYPE POLYBENCH_4D(err_in,NN,ND,IH,IW,nn,nd,ih,iw),
		DATA_TYPE POLYBENCH_4D(err_out,NN,ND,OH,OW,nn,nd,oh,ow))
{

	int n, d, r, c, h, w, row_st, row_end, col_st, col_nd;
/* Copyright (C) 1991-2015 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses ISO/IEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0.  */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NC >= 1) && (_PB_ND >= 1) && (_PB_NN >= 1) && (_PB_NR >= 1) && (dh >= 1) && (dw >= 1) && (ih >= 1) && (iw >= 1)) {
  lbp=0;
  ubp=floord(_PB_NN-1,32);
#pragma omp parallel for private(lbv,ubv,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11)
  for (t1=lbp;t1<=ubp;t1++) {
    for (t2=0;t2<=floord(_PB_ND-1,32);t2++) {
      for (t3=0;t3<=min(floord(ih-1,32),floord(10*_PB_NR+dh-11,32));t3++) {
        for (t4=0;t4<=min(floord(iw-1,32),floord(10*_PB_NC+dw-11,32));t4++) {
          for (t5=max(0,ceild(32*t3-dh-309,320));t5<=min(floord(t3,10),floord(_PB_NR-1,32));t5++) {
            for (t6=32*t1;t6<=min(_PB_NN-1,32*t1+31);t6++) {
              for (t7=32*t2;t7<=min(_PB_ND-1,32*t2+31);t7++) {
                for (t8=32*t3;t8<=min(min(min(ih-1,32*t3+31),10*_PB_NR+dh-11),320*t5+dh+309);t8++) {
                  for (t9=32*t4;t9<=min(min(iw-1,32*t4+31),10*_PB_NC+dw-11);t9++) {
                    for (t10=max(ceild(t8-dh+1,10),32*t5);t10<=min(min(floord(t8,10),_PB_NR-1),32*t5+31);t10++) {
                      for (t11=max(0,ceild(t9-dw+1,10));t11<=min(floord(t9,10),_PB_NC-1);t11++) {
                        err_in[t6][t7][t8][t9] += err_out[t6][t7][t10][t11];;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/* End of CLooG code */
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
	sumpool2d_forward(nn, nd, ih, iw, oh ,ow, dh, dw, sh, sw,
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(out_F));

	sumpool2d_backward(nn, nd, ih, iw, oh ,ow, dh, dw, sh, sw,
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
