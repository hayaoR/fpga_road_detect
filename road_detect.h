#ifndef _ROAD_DETECT_COFIG_
#define _ROAD_DETECT_COFIG_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.h"


#define NPC1 XF_NPPC1

#define TYPE XF_8UC1

struct axis_t {
    ap_uint<8> data;
    ap_int<1> last;
};

struct out_t {
    ap_uint<16> data;
    ap_int<1> last;
};

#define HEIGHT 480
#define WIDTH 640
#define SIZE 9

#define P_HEIGHT 92
#define P_WIDTH 1601
#define P_TYPE XF_8UC1

typedef ap_uint<8> pixel;
typedef ap_fixed<16,2,AP_RND> accuracy;
typedef ap_uint<13> acc;

struct v_pixels {
	pixel upper;
	pixel middle;
	pixel bottom;
};


pixel mean_threshold(pixel window[3][3]);
void adptive_threshold(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &in, xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &out);

void hough (axis_t *src, axis_t *dst, int src_rows, int src_cols, int dst_rows, int dst_cols);
void median_blur(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &in, xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &out);
pixel median_filter(pixel window[3][3]);
pixel median(pixel in[SIZE]);

v_pixels sort3 (pixel y0, pixel y1, pixel y2);
pixel min3(pixel x0, pixel x1, pixel x2);
pixel med(pixel x3, pixel x4, pixel x5);
pixel max3(pixel x6, pixel x7, pixel x8);

#endif
