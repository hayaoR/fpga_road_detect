/*
 	 	 0~30, 180~150の直線を検出できるように書いてみた。
 */

#include "road_detect.h"
#include "hls_math.h"

#define pai 3.1415926
#define angle 60
#define rho 1000


void adptive_threshold(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &in, xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &out) {

	pixel window[3][3];
	pixel line_buffer[2][WIDTH];

#pragma HLS array_partition variable=line_buffer complete dim=1
#pragma HLS array_partition variable=window complete dim=0
	row_loop: for(int row=0; row < HEIGHT+1; row++) {
		col_loop: for(int col=0; col < WIDTH+1; col++) {
#pragma HLS pipeline II=1
			pixel p;
			if (row < HEIGHT && col < WIDTH) {
				p = in.data[row*WIDTH+col];
			}
			for(int i = 0; i < 3; i++) {
				window[i][0] = window[i][1];
				window[i][1] = window[i][2];
			}
			if (col < WIDTH) {
				window[0][2] = (line_buffer[0][col]);
				window[1][2] = (line_buffer[0][col] = line_buffer[1][col]);
				window[2][2] = (line_buffer[1][col] = p);

			}

			if (row >= 1 && col >= 1) {
				int outrow = row - 1;
				int outcol = col -1;
				if(outrow == 0 || outcol == 0
						|| outrow == (HEIGHT - 1) || outcol == (WIDTH - 1)) {
					out.data[outrow*WIDTH+outcol] = 0;

				} else {
					out.data[outrow*WIDTH+outcol] = mean_threshold(window);
				}

			}
		}
	}
}

pixel mean_threshold(pixel window[3][3]) {
	ap_uint<32>sum=0;
	pixel out;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			sum += window[i][j];
		}
	}

	if ((sum - 3*9) > window[1][1]*9) {
		out = 255;
	} else {
		out = 0;
	}
	return out;
}

void median_blur(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &in, xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &out) {

	pixel window[3][3];
	pixel line_buffer[2][WIDTH];
#pragma HLS array_partition variable=line_buffer complete dim=1
#pragma HLS array_partition variable=window complete dim=0
	row_loop: for(int row=0; row < HEIGHT+1; row++) {
		col_loop: for(int col=0; col < WIDTH+1; col++) {
#pragma HLS pipeline II=1
			pixel p;
			if (row < HEIGHT && col < WIDTH) {
				p = in.data[row*WIDTH+col];
			}
			for(int i = 0; i < 3; i++) {
				window[i][0] = window[i][1];
				window[i][1] = window[i][2];
			}
			if (col < WIDTH) {
				window[0][2] = (line_buffer[0][col]);
				window[1][2] = (line_buffer[0][col] = line_buffer[1][col]);
				window[2][2] = (line_buffer[1][col] = p);

			}

			if (row >= 1 && col >= 1) {
				int outrow = row - 1;
				int outcol = col -1;
				if(outrow == 0 || outcol == 0 ||
					outrow == (HEIGHT - 1) ||
					outcol == (WIDTH - 1)) {
					out.data[outrow*WIDTH+outcol] = 0;
				} else {
					out.data[outrow*WIDTH+outcol] = median_filter(window);
				}
			}
		}
	}
}

pixel median_filter(pixel window[3][3]) {
	v_pixels tmp[3];
#pragma HLS array_partition variable=tmp complete
	pixel t0, t1, t2;
	pixel out;
	for (int i = 0; i < 3; i++) {
#pragma HLS unroll
		tmp[i] = sort3(window[0][i], window[1][i], window[2][i]);
	}

#pragma HLS DATAFLOW
	t0 = min3(tmp[0].upper, tmp[1].upper, tmp[1].upper);
	t1 = med(tmp[0].middle, tmp[1].middle, tmp[2].middle);
	t2 = max3(tmp[0].bottom, tmp[1].bottom, tmp[2].bottom);
	out = med(t0, t1, t2);

	return out;

}

v_pixels sort3 (pixel y0, pixel y1, pixel y2) {
	pixel t0, t1;
	pixel tt1;
	pixel out0, out1, out2;

	v_pixels out;

	if (y0 > y1) {
		t0 = y0;
		t1 = y1;
	} else {
		t0 = y1;
		t1 = y0;
	}

	if (t1 > y2) {
		tt1 = t1;
		out2 = y2;
	} else {
		tt1 = y2;
		out2 = t1;
	}

	if (t0 > tt1) {
		out0 = t0;
		out1 = tt1;
	} else {
		out0 = tt1;
		out1 = t0;
	}

	out.upper = out0;
	out.middle = out1;
	out.bottom = out2;

	return out;
}

pixel min3(pixel x0, pixel x1, pixel x2) {
	pixel t1;
	pixel out2;

	t1 = (x0 < x1) ? x0 : x1;
	out2 = (t1 < x2) ? t1 : x2;

	return out2;
}

pixel med(pixel x3, pixel x4, pixel x5) {
	pixel t3, t4;
	pixel tt4;
	pixel out4;

	if (x3 > x4) {
		t3 = x3;
		t4 = x4;
	} else {
		t3 = x4;
		t4 = x3;
	}

	tt4 = (t4 > x5) ? t4 : x5;

	out4 = (t3 < tt4) ? t3 : tt4;

	return out4;
}

pixel max3(pixel x6, pixel x7, pixel x8) {
	pixel t7;
	pixel out6;

	t7 = (x7 < x8)? x8 : x7;
	out6 = (x6 < t7)? t7 : x6;

	return out6;
}

void full_accum(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &in,acc accum[angle][rho]) {

  //      accuracy cosval[angle/2] = {1.0, 0.9998476951563913, 0.9993908270190958, 0.9986295347545738, 0.9975640502598242, 0.9961946980917455, 0.9945218953682733, 0.992546151641322, 0.9902680687415704, 0.9876883405951378, 0.984807753012208, 0.981627183447664, 0.9781476007338057, 0.9743700647852352, 0.9702957262759965, 0.9659258262890683, 0.9612616959383189, 0.9563047559630354, 0.9510565162951535, 0.9455185755993168, 0.9396926207859084, 0.9335804264972017, 0.9271838545667874, 0.9205048534524404, 0.9135454576426009, 0.9063077870366499, 0.898794046299167, 0.8910065241883679, 0.882947592858927, 0.8746197071393957};
;
   //     accuracy sinval[angle/2] = {0.0, 0.01745240643728351, 0.03489949670250097, 0.05233595624294383, 0.0697564737441253, 0.08715574274765817, 0.10452846326765346, 0.12186934340514748, 0.13917310096006544, 0.15643446504023087, 0.17364817766693033, 0.1908089953765448, 0.20791169081775931, 0.224951054343865, 0.24192189559966773, 0.25881904510252074, 0.27563735581699916, 0.29237170472273677, 0.3090169943749474, 0.32556815445715664, 0.3420201433256687, 0.35836794954530027, 0.374606593415912, 0.3907311284892737, 0.40673664307580015, 0.42261826174069944, 0.4383711467890774, 0.45399049973954675, 0.4694715627858908, 0.48480962024633706};
;

			accuracy cosval[angle/2];
			accuracy sinval[angle/2];
#pragma HLS ARRAY_PARTITION variable=sinval complete dim=0
#pragma HLS ARRAY_PARTITION variable=cosval complete dim=0
        accuracy Angle_accuracy=pai/180;

        acc addr[angle];
        acc accbuf[2][angle];
#pragma HLS ARRAY_PARTITION variable=addr complete dim=0
#pragma HLS ARRAY_PARTITION variable=accbuf complete dim=0

        ap_fixed<14,13> t1, t2;


 for(int i=0;i<angle/2;i++)
        {
            sinval[i]=::hls::sinf(i*Angle_accuracy);
            cosval[i]=::hls::cosf(i*Angle_accuracy);
        }

 loop_init_r: for(int r=0;r<rho;r++)
     {
     loop_init_n: for(int n=0;n<angle;n++)
         {
 #pragma HLS PIPELINE
             accum[n][r]=0;
         }
     }

  loop_init: for(int n = 0; n < angle; n++ )
     {
         addr[n]=0;
         accbuf[0][n]=0;
     }

  loop_height: for( int i = 0; i < HEIGHT; i++ )
     {
     loop_width: for( int  j = 0; j < WIDTH; j++ )
         {
 #pragma HLS PIPELINE
 #pragma HLS DEPENDENCE array inter false
             if(in.data[i*WIDTH+j]!=0)
             {
             loop_angle: for(int n = 0; n < angle/2; n++ )
                 {
            	 	 		int n2 = n + angle/2;
                     accbuf[1][n]=accbuf[0][n];
                     accbuf[1][n2]=accbuf[0][n2];

                     t1=j*cosval[n]+i*sinval[n];
                     t2=-j*cosval[n]+i*sinval[n];

                     acc r1 = t1.range(13,1);
                     acc r2 = t2.range(13,1)+700;

                     accbuf[0][n]=accum[n][r1];
                     accbuf[0][n2]=accum[n2][r2];

                   if(r1==addr[n])
                         accbuf[0][n]=accbuf[0][n]+1;
                   if(r2==addr[n2])
                	   	   accbuf[0][n2]=accbuf[0][n2]+1;

                     accum[n][addr[n]]=accbuf[1][n]+1;
                     accum[n2][addr[n2]]=accbuf[1][n2]+1;

                     addr[n]=r1;
                     addr[n2]=r2;
                 }
             }

         }
     }
  loop_exit: for(int n = 0; n < angle/2; n++ )
     {
	  	  int n2 = n + angle/2;
         accum[n][addr[n]]=accbuf[0][n]+1;
         accum[n2][addr[n2]]=accbuf[0][n2]+1;
     }
}


void axis2xfMat (xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_src, axis_t *src, int src_rows, int src_cols) {
#pragma HLS inline off

	for (int i=0; i<src_rows; i++) {
		for (int j=0; j<src_cols; j++) {
#pragma HLS pipeline
#pragma HLS loop_flatten off
			_src.data[i*src_cols+j] = src[i*src_cols+j].data;
		}	
	}

}

void xfMat2axis (xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_dst, axis_t *dst, int dst_rows, int dst_cols) {
#pragma HLS inline off

	for (int i=0; i<dst_rows; i++) {
		for (int j=0; j<dst_cols; j++) {
#pragma HLS pipeline
#pragma HLS loop_flatten off
			ap_uint<1> tmp = 0;
			if ((i==dst_rows-1) && (j== dst_cols-1)) {
				tmp = 1;
			}
			dst[i*dst_cols+j].last = tmp;
			dst[i*dst_cols+j].data = _dst.data[i*dst_cols+j];
		}
	}
}

void P_xfMat2axis (acc accum[angle][rho], out_t *dst, int dst_rows, int dst_cols) {
#pragma HLS inline off

	for (int i=0; i<dst_rows; i++) {
		for (int j=0; j<dst_cols; j++) {
#pragma HLS pipeline
#pragma HLS loop_flatten off
			ap_uint<1> tmp = 0;
			if ((i==dst_rows-1) && (j== dst_cols-1)) {
				tmp = 1;
			}
			dst[i*dst_cols+j].last = tmp;
			dst[i*dst_cols+j].data = accum[i][j];
		}
	}
}

void hough(axis_t *src, out_t *dst,int src_rows, int src_cols, int dst_rows, int dst_cols) {
	
#pragma HLS INTERFACE axis port=src depth=384*288 // Added depth for C/RTL cosimulation
#pragma HLS INTERFACE axis port=dst depth=192*144 // Added depth for C/RTL cosimulation
#pragma HLS INTERFACE s_axilite port=src_rows
#pragma HLS INTERFACE s_axilite port=src_cols
#pragma HLS INTERFACE s_axilite port=dst_rows
#pragma HLS INTERFACE s_axilite port=dst_cols
#pragma HLS INTERFACE s_axilite port=return

	xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> _src(HEIGHT, WIDTH);
	xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> _img0(HEIGHT, WIDTH);
	xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> _img1(HEIGHT, WIDTH);

#pragma HLS stream variable=_src.data depth=150
#pragma HLS stream variable=_img0.data depth=150
#pragma HLS stream variable=_img1.data depth=150


	acc _accum[angle][rho];
#pragma HLS ARRAY_PARTITION variable= _accum complete dim=1

#pragma HLS dataflow
	
	axis2xfMat(_src, src, HEIGHT, WIDTH);

	median_blur(_src, _img0);
	adptive_threshold(_img0, _img1);
	full_accum(_img1, _accum);

	P_xfMat2axis(_accum, dst, angle, rho);
	//xfMat2axis(_dst, dst, dst_rows, dst_cols);
	//xfMat2axis(_src, dst, dst_rows, dst_cols);
}

