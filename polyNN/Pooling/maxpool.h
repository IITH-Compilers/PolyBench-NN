/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _MAXPOOL_H
# define _MAXPOOL_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

# if !defined(NN) && !defined(ND) && !defined(IH) && !defined(IW) && !defined(DH) && !defined(DW) && !defined(SH) && !defined(SW) && !defined(OH) && !defined(OW) 
/* Define sample dataset sizes. */
#  ifdef MINI_DATASET
#   define SH 2
#   define SW 2
#   define DH 3
#   define DW 3
#   define IH 10
#   define IW 10
#   define ND 20
#   define NN 30
#  endif 

#  ifdef SMALL_DATASET
#   define SH 5
#   define SW 5
#   define DH 7
#   define DW 7
#   define IH 30
#   define IW 30
#   define ND 50
#   define NN 70
#  endif 

#  ifdef MEDIUM_DATASET
#   define SH 7
#   define SW 7
#   define DH 10
#   define DW 10
#   define IH 75
#   define IW 75
#   define ND 90
#   define NN 120
#  endif 

#  ifdef LARGE_DATASET
#   define SH 10
#   define SW 10
#   define DH 15
#   define DW 15
#   define IH 100
#   define IW 100
#   define ND 120
#   define NN 150
#  endif 

#  ifdef EXTRALARGE_DATASET
#   define SH 15
#   define SW 15
#   define DH 20
#   define DW 20
#   define IH 200
#   define IW 200
#   define ND 210
#   define NN 250
#  endif 


#endif /* !(SH SW DH DW IH IW ND NN) */

#define OH (IH - DH)/SH + 1
#define OW (IW - DW)/SW + 1

# define _PB_NN POLYBENCH_LOOP_BOUND(NN,nn)
# define _PB_ND POLYBENCH_LOOP_BOUND(ND,nd)
# define _PB_IH POLYBENCH_LOOP_BOUND(IH,ih)
# define _PB_IW POLYBENCH_LOOP_BOUND(IW,iw)
# define _PB_DH POLYBENCH_LOOP_BOUND(DH,dh)
# define _PB_DW POLYBENCH_LOOP_BOUND(DW,dw)
# define _PB_SH POLYBENCH_LOOP_BOUND(SH,sh)
# define _PB_SW POLYBENCH_LOOP_BOUND(SW,sw)
# define _PB_OH POLYBENCH_LOOP_BOUND(OH,oh)
# define _PB_OW POLYBENCH_LOOP_BOUND(OW,ow)
# define _PB_NR POLYBENCH_LOOP_BOUND(OH,oh)
# define _PB_NC POLYBENCH_LOOP_BOUND(OW,ow)

/* Default data type */
# if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#  define DATA_TYPE_IS_DOUBLE
# endif

#ifdef DATA_TYPE_IS_INT
#  define DATA_TYPE int
#  define DATA_PRINTF_MODIFIER "%d "
#endif 

#ifdef DATA_TYPE_IS_FLOAT
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
#  define SCALAR_VAL(x) x##f
#  define SQRT_FUN(x) sqrtf(x)
#  define EXP_FUN(x) expf(x)
#  define POW_FUN(x,y) powf(x,y)
# endif

#ifdef DATA_TYPE_IS_DOUBLE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
#  define SCALAR_VAL(x) x
#  define SQRT_FUN(x) sqrt(x)
#  define EXP_FUN(x) exp(x)
#  define POW_FUN(x,y) pow(x,y)
# endif

#endif /* !_MAXPOOL_H */

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
