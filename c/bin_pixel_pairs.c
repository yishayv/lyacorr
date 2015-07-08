/* Example of wrapping the cos function from math.h using the Numpy-C-API. */

/*
 * this is a slightly modified version of:
 * https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/* #define _MY_DEBUG_PRINT */

#ifdef _MY_DEBUG_PRINT
#define MY_DEBUG_PRINT(...) PySys_WriteStdout(__VA_ARGS__);
#else
#define MY_DEBUG_PRINT(...)
#endif

#define SWAP(type, a_, b_) \
do { \
    struct { type *a; type *b; type t; } SWAP; \
    SWAP.a  = &(a_); \
    SWAP.b  = &(b_); \
    SWAP.t  = *SWAP.a; \
    *SWAP.a = *SWAP.b; \
    *SWAP.b = SWAP.t; \
} while (0)

/* based on macros from the numpy API, extended for 5D. */
#define PyArray_GETPTR5(obj, i, j, k, l, m) ((void *)(PyArray_BYTES(obj) + \
                                            (i)*PyArray_STRIDES(obj)[0] + \
                                            (j)*PyArray_STRIDES(obj)[1] + \
                                            (k)*PyArray_STRIDES(obj)[2] + \
                                            (l)*PyArray_STRIDES(obj)[3] + \
                                            (m)*PyArray_STRIDES(obj)[4]))

static inline double get_bin_x(double dist1, double dist2, double x_scale)
{
	/* keep the result as double so that a boundary check can be done */
	/* r|| = abs(r1 - r2) */
	return fabs(dist1 - dist2) * x_scale;
}

static inline double get_bin_y(double dist1, double dist2, double y_scale)
{
	/* keep the result as double so that a boundary check can be done */
	/* r_ = (r1 + r2)/2 * qso_angle */
	return (dist1 + dist2) * y_scale;
}

long find_largest_index(double max_dist_for_qso_angle, PyArrayObject * in_array_dist, long dist_size)
{
	/*
	 * find the largest index of dist2 for which a transverse distance to the other
	 * QSO is within range.
	 */
	double dist;
	long j;

	for (j = 0; j < dist_size; j++)
	{
		dist = *((double *)PyArray_GETPTR1(in_array_dist, j));
		if (dist > max_dist_for_qso_angle)
		{
			return j + 1;
		}
	}
	/* got to the end of the array. simply return the size of the array. */
	return dist_size;
}

int compare_array_shape(PyArrayObject * array, long ndim, long * shape)
{
	int i;
	if (PyArray_NDIM(array) != ndim)
		return 0;
	for (i=0; i < ndim; i++)
	{
		if (PyArray_DIM(array, i) != shape[i])
			return 0;
	}
	return 1;
}

static void
bin_pixel_pairs_loop(PyArrayObject * in_array_dist1,
		     PyArrayObject * in_array_dist2,
		     PyArrayObject * in_array_flux1,
		     PyArrayObject * in_array_flux2,
		     PyArrayObject * in_array_weights1,
		     PyArrayObject * in_array_weights2,
		     PyArrayObject * out_array, double qso_angle,
		     double x_bin_size, double y_bin_size, double x_bin_count, double y_bin_count)
{
	long i, j;
	long dist1_size, dist2_size;
	int bin_x, bin_y;
	long last_dist2_start, first_pair_dist2;
	long max_dist1_index, max_dist2_index;
	double f_bin_x, f_bin_y;
	double dist1, dist2, flux1, flux2, weight1, weight2;
	double *p_current_bin_flux, *p_current_bin_weight, *p_current_bin_count;
	double weighted_flux1, weighted_flux2;
	double x_scale, y_scale;
	double max_dist_for_qso_angle;

	/* iterate over the arrays */
	dist1_size = PyArray_DIM(in_array_dist1, 0);
	dist2_size = PyArray_DIM(in_array_dist2, 0);

	if (!dist1_size || !dist2_size)
		return;

	/* if dist1 is nearer, it is more efficient to run the function with 1 and 2 reversed. */
	dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, 0));
	dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, 0));
	if (dist1 < dist2)
	{
		MY_DEBUG_PRINT("SWAPPING 1,2\n");
		SWAP(double, dist1, dist2);
		SWAP(long, dist1_size, dist2_size);
		SWAP(PyArrayObject *, in_array_dist1, in_array_dist2);
		SWAP(PyArrayObject *, in_array_flux1, in_array_flux2);
		SWAP(PyArrayObject *, in_array_weights1, in_array_weights2);
	}

	/* r|| = abs(r1 - r2) */
	x_scale = 1. / x_bin_size;
	/* r_ = (r1 + r2)/2 * qso_angle */
	y_scale = qso_angle / (2. * y_bin_size);

	/*
	 * find the largest index of dist2 for which a transverse distance to the other
	 * QSO is within range.
	 */
	/* set initial index to the end of the array. */
	max_dist_for_qso_angle = (y_bin_count + 1) * y_bin_size / sin(qso_angle);
	max_dist1_index = find_largest_index(max_dist_for_qso_angle, in_array_dist1, dist1_size);
	max_dist2_index = find_largest_index(max_dist_for_qso_angle, in_array_dist2, dist2_size);

	MY_DEBUG_PRINT("max_dist1_index: %ld, dist1_size: %ld\n", max_dist1_index, dist1_size);
	MY_DEBUG_PRINT("max_dist2_index: %ld, dist2_size: %ld\n", max_dist2_index, dist2_size);

	MY_DEBUG_PRINT(":::::Before loop\n");

	last_dist2_start = 0;
	for (i = 0; i < max_dist1_index; i++)
	{
		/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i); */
		dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, i));
		flux1 = *((double *)PyArray_GETPTR1(in_array_flux1, i));
		weight1 = *((double *)PyArray_GETPTR1(in_array_weights1, i));

		weighted_flux1 = flux1 * weight1;

		/*
		 * distance values are ordered, so if any dist2 was too low to be close enough to the previous dist1,
		 * the same should hold for the current dist1.
		 */
		first_pair_dist2 = 0;
		for (j = last_dist2_start; j < max_dist2_index; j++)
		{
			dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, j));
			flux2 = *((double *)PyArray_GETPTR1(in_array_flux2, j));
			weight2 = *((double *)PyArray_GETPTR1(in_array_weights2, j));

			f_bin_x = get_bin_x(dist1, dist2, x_scale);
			f_bin_y = get_bin_y(dist1, dist2, y_scale);

			if (f_bin_x >= 0 && f_bin_x < x_bin_count)
			{
				/* pixel is in range of parallel separation */
				if (!first_pair_dist2)
					first_pair_dist2 = j;

				if (f_bin_y >= 0 && f_bin_y < y_bin_count)
				{
					/* pixel is in range */
					
					weighted_flux2 = flux2 * weight2;
					
					bin_x = f_bin_x;
					bin_y = f_bin_y;
					
					p_current_bin_flux = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, 0);
					(*p_current_bin_flux) += weighted_flux1 * weighted_flux2;
					p_current_bin_count = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, 1);
					(*p_current_bin_count) += 1;
					p_current_bin_weight = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, 2);
					(*p_current_bin_weight) += weight1 * weight2;
				}
			} else
			{
				/*
				 * in flat geometry we cannot move in and out of range more than once.
				 */
				if (first_pair_dist2)
					break;
			}
		}
		if (first_pair_dist2)
			last_dist2_start = first_pair_dist2;
	}
}

/* wrapped function */
static PyObject *bin_pixel_pairs(PyObject * self, PyObject * args, PyObject * kw)
{

	PyArrayObject *in_array_dist1;
	PyArrayObject *in_array_dist2;
	PyArrayObject *in_array_flux1;
	PyArrayObject *in_array_flux2;
	PyArrayObject *in_array_weights1;
	PyArrayObject *in_array_weights2;
	PyArrayObject *out_array;
	double qso_angle;
	double x_bin_size, y_bin_size;
	double x_bin_count, y_bin_count;
	npy_intp out_dim[3] = { 0 };

	static char *kwlist[] = { "ar_dist1", "ar_dist2",
		"ar_flux1", "ar_flux2", "ar_weights1", "ar_weights2",
		"qso_angle", "x_bin_size", "y_bin_size", "x_bin_count", "y_bin_count",
		NULL
	};

	MY_DEBUG_PRINT(":::::Function Start\n");

	/* parse numpy array arguments */
	if (!PyArg_ParseTupleAndKeywords
	    (args, kw, "O!O!O!O!O!O!ddddd:bin_pixel_pairs", kwlist,
	     &PyArray_Type, &in_array_dist1, &PyArray_Type, &in_array_dist2,
	     &PyArray_Type, &in_array_flux1, &PyArray_Type, &in_array_flux2,
	     &PyArray_Type, &in_array_weights1, &PyArray_Type, &in_array_weights2,
	     &qso_angle, &x_bin_size, &y_bin_size, &x_bin_count, &y_bin_count))
	{
		return NULL;
	}
	MY_DEBUG_PRINT(":::::After arg parse\n");

	/*
	 * construct a 3D output array, (x_bin_count) by (y_bin_count) by (flux, counts, weights)
	 */
	out_dim[0] = x_bin_count;
	out_dim[1] = y_bin_count;
	out_dim[2] = 3;
	out_array = (PyArrayObject *) PyArray_ZEROS(3, out_dim, NPY_DOUBLE, 0);
	if (out_array == NULL)
		return NULL;

	MY_DEBUG_PRINT(":::::Created out array\n");

	bin_pixel_pairs_loop(in_array_dist1, in_array_dist2,
			     in_array_flux1, in_array_flux2,
			     in_array_weights1, in_array_weights2,
			     out_array, qso_angle, x_bin_size, y_bin_size, x_bin_count, y_bin_count);

	return (PyObject *) out_array;
}

static void
bin_pixel_pairs_histogram_loop(PyArrayObject * in_array_dist1,
			       PyArrayObject * in_array_dist2,
			       PyArrayObject * in_array_flux1,
			       PyArrayObject * in_array_flux2,
			       PyArrayObject * in_array_weights1,
			       PyArrayObject * in_array_weights2,
			       PyArrayObject * out_array, double qso_angle,
			       double x_bin_size, double y_bin_size,
			       double x_bin_count, double y_bin_count,
			       double f_min, double f_max, double f_bin_count, double *p_pair_count)
{
	long i, j;
	long dist1_size, dist2_size;
	int bin_x, bin_y, bin_f;
	long last_dist2_start, first_pair_dist2;
	double dist1, dist2, flux1, flux2, weight1, weight2;
	double *p_current_bin_flux;
	double flux_product;
	double x_scale, y_scale;

	/* iterate over the arrays */
	dist1_size = PyArray_DIM(in_array_dist1, 0);
	dist2_size = PyArray_DIM(in_array_dist2, 0);

	x_scale = 1. / x_bin_size;
	y_scale = qso_angle / (2. * y_bin_size);

	MY_DEBUG_PRINT(":::::Before loop\n");

	dist1 = 1;
	last_dist2_start = 0;
	for (i = 0; i < dist1_size && dist1; i++)
	{
		/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i); */

		dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, i));
		flux1 = *((double *)PyArray_GETPTR1(in_array_flux1, i));
		weight1 = *((double *)PyArray_GETPTR1(in_array_weights1, i));

		dist2 = 1;
		/*
		 * distance values are ordered, so if any dist2 was too low to be close enough to the previous dist1,
		 * the same should hold for the current dist1.
		 */
		first_pair_dist2 = 0;
		for (j = last_dist2_start; j < dist2_size && dist2; j++)
		{
			dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, j));
			flux2 = *((double *)PyArray_GETPTR1(in_array_flux2, j));
			weight2 = *((double *)PyArray_GETPTR1(in_array_weights2, j));

			/* r|| = abs(r1 - r2) */
			bin_x = (int)(fabs(dist1 - dist2) * x_scale);
			/* r_ = (r1 + r2)/2 * qso_angle */
			bin_y = (int)((dist1 + dist2) * y_scale);

			if ((bin_x < x_bin_count) && (bin_y < y_bin_count))
			{
				/* pixel is in range */
				if (!first_pair_dist2)
					first_pair_dist2 = j;

				/* find the correct bin for the flux product */
				flux_product = flux1 * flux2;

				/* if the value is out of range, we must add it to the minimum/maximum 
				 * to preserve the quantile value of each intermediate bin.
				 */
				if (flux_product > f_max)
					bin_f = f_bin_count - 1;
				else if (flux_product < f_min)
					bin_f = 0;
				else
					bin_f = (int)(f_bin_count * (flux_product - f_min) / (f_max - f_min));

				p_current_bin_flux = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, bin_f);
				/* MY_DEBUG_PRINT(":::::Adding pixel, %d, %d, %d, weight:%lf\n", 
				   bin_x, bin_y, bin_f, weight1*weight2); */
				(*p_current_bin_flux) += weight1 * weight2;
				(*p_pair_count)++;
			} else
			{
				/*
				 * in flat geometry we cannot move in and out of range more than once.
				 */
				if (first_pair_dist2)
					break;
			}
		}
		if (first_pair_dist2)
			last_dist2_start = first_pair_dist2;
	}
}

/* wrapped function */
static PyObject *bin_pixel_pairs_histogram(PyObject * self, PyObject * args, PyObject * kw)
{

	PyArrayObject *in_array_dist1;
	PyArrayObject *in_array_dist2;
	PyArrayObject *in_array_flux1;
	PyArrayObject *in_array_flux2;
	PyArrayObject *in_array_weights1;
	PyArrayObject *in_array_weights2;
	PyArrayObject *out_array;
	double qso_angle;
	double x_bin_size, y_bin_size;
	double x_bin_count, y_bin_count, f_bin_count;
	double f_min, f_max;
	double pair_count;
	PyObject *ret_val;

	static char *kwlist[] = { "ar_dist1", "ar_dist2",
		"ar_flux1", "ar_flux2", "ar_weights1", "ar_weights2",
		"out",
		"qso_angle", "x_bin_size", "y_bin_size", "x_bin_count", "y_bin_count",
		"f_min", "f_max", "f_bin_count", "pair_count",
		NULL
	};

	MY_DEBUG_PRINT(":::::Function Start\n");

	/* parse numpy array arguments */
	if (!PyArg_ParseTupleAndKeywords
	    (args, kw, "O!O!O!O!O!O!O!ddddddddd:bin_pixel_pairs_histogram", kwlist,
	     &PyArray_Type, &in_array_dist1, &PyArray_Type, &in_array_dist2,
	     &PyArray_Type, &in_array_flux1, &PyArray_Type, &in_array_flux2,
	     &PyArray_Type, &in_array_weights1, &PyArray_Type, &in_array_weights2,
	     &PyArray_Type, &out_array,
	     &qso_angle, &x_bin_size, &y_bin_size, &x_bin_count, &y_bin_count, &f_min, &f_max, &f_bin_count,
	     &pair_count))
	{
		return NULL;
	}
	MY_DEBUG_PRINT(":::::After arg parse\n");

	bin_pixel_pairs_histogram_loop(in_array_dist1, in_array_dist2,
				       in_array_flux1, in_array_flux2,
				       in_array_weights1, in_array_weights2,
				       out_array, qso_angle,
				       x_bin_size, y_bin_size, x_bin_count, y_bin_count, f_min, f_max, f_bin_count,
				       &pair_count);

	ret_val = PyFloat_FromDouble(pair_count);
	return ret_val;
}

static void
bin_pixel_quads_loop(PyArrayObject * in_array_dist1,
		     PyArrayObject * in_array_dist2,
		     PyArrayObject * in_array_flux1,
		     PyArrayObject * in_array_flux2,
		     PyArrayObject * in_array_weights1,
		     PyArrayObject * in_array_weights2,
		     PyArrayObject * in_array_dist3,
		     PyArrayObject * in_array_dist4,
		     PyArrayObject * in_array_flux3,
		     PyArrayObject * in_array_flux4,
		     PyArrayObject * in_array_weights3,
		     PyArrayObject * in_array_weights4,
		     PyArrayObject * in_array_estimator,
		     PyArrayObject * out_array,
		     double qso_angle12, double qso_angle34,
		     double x_bin_size, double y_bin_size, double x_bin_count, double y_bin_count)
{
	long i, j, k, l;
	long dist1_size, dist2_size, dist3_size, dist4_size;
	int bin_x_a, bin_y_a, bin_x_b, bin_y_b;
	long last_dist2_start, first_pair_dist2, last_dist4_start, first_pair_dist4;
	long max_dist1_index, max_dist2_index, max_dist3_index, max_dist4_index;
	double dist1, dist2, dist3, dist4;
	double flux1, flux2, flux3, flux4;
	double weight1, weight2, weight3, weight4;
	double weighted_flux1, weighted_flux2, weighted_flux3, weighted_flux4;
	double *p_current_bin_flux, *p_current_bin_weight, *p_current_bin_count;
	double x_scale, y_scale12, y_scale34;
	double max_dist_for_qso_angle12, max_dist_for_qso_angle34;
	double est_12, est_34, cov_term_12, cov_term_34;
	long estimator_shape[2], out_array_shape[5];

	MY_DEBUG_PRINT("Inside bin_pixels_quad_loop()\n");
	
	/* get array sizes */
	dist1_size = PyArray_DIM(in_array_dist1, 0);
	dist2_size = PyArray_DIM(in_array_dist2, 0);
	dist3_size = PyArray_DIM(in_array_dist3, 0);
	dist4_size = PyArray_DIM(in_array_dist4, 0);
	
	/* verify that the estimator array has the correct shape */
	estimator_shape[0] = x_bin_count;
	estimator_shape[1] = y_bin_count;
	if (!compare_array_shape(in_array_estimator, 2, estimator_shape))
		return;
	
	/* verify that the output array has the correct shape */
	out_array_shape[0] = x_bin_count;
	out_array_shape[1] = y_bin_count;
	out_array_shape[2] = x_bin_count;
	out_array_shape[3] = y_bin_count;
	out_array_shape[4] = 3;
	if (!compare_array_shape(out_array, 5, out_array_shape))
		return;
	
	if (!dist1_size || !dist2_size || !dist3_size || !dist4_size)
		return;

	/* if dist1 is nearer, it is more efficient to run the loops with 1 and 2 reversed. */
	dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, 0));
	dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, 0));
	dist3 = *((double *)PyArray_GETPTR1(in_array_dist3, 0));
	dist4 = *((double *)PyArray_GETPTR1(in_array_dist4, 0));
	if (dist1 < dist2)
	{
		SWAP(double, dist1, dist2);
		SWAP(long, dist1_size, dist2_size);
		SWAP(PyArrayObject *, in_array_dist1, in_array_dist2);
		SWAP(PyArrayObject *, in_array_flux1, in_array_flux2);
		SWAP(PyArrayObject *, in_array_weights1, in_array_weights2);
	}
	/* the same applies to 3 and 4. */
	if (dist3 < dist4)
	{
		SWAP(double, dist3, dist4);
		SWAP(long, dist3_size, dist4_size);
		SWAP(PyArrayObject *, in_array_dist3, in_array_dist4);
		SWAP(PyArrayObject *, in_array_flux3, in_array_flux4);
		SWAP(PyArrayObject *, in_array_weights3, in_array_weights4);
	}

	x_scale = 1. / x_bin_size;
	y_scale12 = qso_angle12 / (2. * y_bin_size);
	y_scale34 = qso_angle34 / (2. * y_bin_size);
	max_dist_for_qso_angle12 = y_bin_count / y_scale12;
	max_dist_for_qso_angle34 = y_bin_count / y_scale34;

	/*
	 * find the largest index of dist1(,2,3,4) for which a transverse distance to the other
	 * QSO is within range.
	 */
	max_dist1_index = find_largest_index(max_dist_for_qso_angle12, in_array_dist1, dist1_size);
	max_dist2_index = find_largest_index(max_dist_for_qso_angle12, in_array_dist2, dist2_size);
	max_dist3_index = find_largest_index(max_dist_for_qso_angle34, in_array_dist3, dist3_size);
	max_dist4_index = find_largest_index(max_dist_for_qso_angle34, in_array_dist4, dist4_size);

	MY_DEBUG_PRINT(":::::Before loop\n");

	last_dist2_start = 0;
	for (i = 0; i < max_dist1_index; i++)
	{
		/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i); */
		dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, i));
		flux1 = *((double *)PyArray_GETPTR1(in_array_flux1, i));
		weight1 = *((double *)PyArray_GETPTR1(in_array_weights1, i));

		weighted_flux1 = flux1 * weight1;

		/*
		 * distance values are ordered, so if any dist2 was too low to be close enough to the previous dist1,
		 * the same should hold for the current dist1.
		 */
		first_pair_dist2 = 0;
		for (j = last_dist2_start; j < max_dist2_index; j++)
		{
			dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, j));
			flux2 = *((double *)PyArray_GETPTR1(in_array_flux2, j));
			weight2 = *((double *)PyArray_GETPTR1(in_array_weights2, j));

			bin_x_a = get_bin_x(dist1, dist2, x_scale);
			bin_y_a = get_bin_y(dist1, dist2, y_scale12);

			weighted_flux2 = flux2 * weight2;
			if ((bin_x_a < x_bin_count) && (bin_y_a < y_bin_count))
			{
				/* pixel is in range */
				if (!first_pair_dist2)
					first_pair_dist2 = j;

				est_12 = *((double*)PyArray_GETPTR2(in_array_estimator, bin_x_a, bin_y_a));
				
				cov_term_12 = (weighted_flux1 * weighted_flux2) * (flux1 * flux2 - est_12);
				
				last_dist4_start = 0;
				for (k = 0; k < max_dist3_index; k++)
				{
					dist3 = *((double *)PyArray_GETPTR1(in_array_dist3, k));
					flux3 = *((double *)PyArray_GETPTR1(in_array_flux3, k));
					weight3 = *((double *)PyArray_GETPTR1(in_array_weights3, k));

					weighted_flux3 = flux3 * weight3;

					first_pair_dist4 = 0;
					for (l = last_dist4_start; l < max_dist4_index; l++)
					{
						dist4 = *((double *)PyArray_GETPTR1(in_array_dist4, l));
						flux4 = *((double *)PyArray_GETPTR1(in_array_flux4, l));
						weight4 = *((double *)PyArray_GETPTR1(in_array_weights4, l));

						bin_x_b = get_bin_x(dist3, dist4, x_scale);
						bin_y_b = get_bin_y(dist3, dist4, y_scale34);
						
						if ((bin_x_b < x_bin_count) && (bin_y_b < y_bin_count))
						{
							weighted_flux4 = flux4 * weight4;

							est_34 = *((double*)PyArray_GETPTR2(in_array_estimator, bin_x_b, bin_y_b));
							
							cov_term_34 = (weighted_flux3 * weighted_flux4) * (flux3 * flux4 - est_12);
							
							p_current_bin_flux =
							    (double *)PyArray_GETPTR5(out_array,
										      bin_x_a, bin_y_a, bin_x_b,
										      bin_y_b, 0);
							(*p_current_bin_flux) +=
								cov_term_12 * cov_term_34;
							
							p_current_bin_count =
							    (double *)PyArray_GETPTR5(out_array, bin_x_a, bin_y_a,
										      bin_x_b, bin_y_b, 1);
							(*p_current_bin_count) += 1;
							p_current_bin_weight =
							    (double *)PyArray_GETPTR5(out_array,
										      bin_x_a, bin_y_a, bin_x_b,
										      bin_y_b, 2);
							(*p_current_bin_weight) +=
							    (weight1 * weight2) * (weight3 * weight4);
						} else
						{
							if (first_pair_dist4)
								break;
						}
					}
					if (first_pair_dist4)
						last_dist4_start = first_pair_dist4;
				}
			} else
			{
				/*
				 * in flat geometry we cannot move in and out of range more than once.
				 */
				if (first_pair_dist2)
					break;
			}
		}
		if (first_pair_dist2)
			last_dist2_start = first_pair_dist2;
	}
}


/* wrapped function */
static PyObject *bin_pixel_quads(PyObject * self, PyObject * args, PyObject * kw)
{
	
	PyArrayObject *in_array_dist1;
	PyArrayObject *in_array_dist2;
	PyArrayObject *in_array_flux1;
	PyArrayObject *in_array_flux2;
	PyArrayObject *in_array_weights1;
	PyArrayObject *in_array_weights2;
	PyArrayObject *in_array_dist3;
	PyArrayObject *in_array_dist4;
	PyArrayObject *in_array_flux3;
	PyArrayObject *in_array_flux4;
	PyArrayObject *in_array_weights3;
	PyArrayObject *in_array_weights4;
	PyArrayObject *in_array_estimator;
	PyArrayObject *out_array;
	double qso_angle12, qso_angle34;
	double x_bin_size, y_bin_size;
	double x_bin_count, y_bin_count;
	
	static char *kwlist[] = { "ar_dist1", "ar_dist2",
		"ar_flux1", "ar_flux2",
		"ar_weights1", "ar_weights2",
		"ar_dist3", "ar_dist4",
		"ar_flux3", "ar_flux4",
		"ar_weights3", "ar_weights4",
		"ar_est",
		"out",
		"qso_angle12", "qso_angle34",
		"x_bin_size", "y_bin_size", "x_bin_count", "y_bin_count",
		NULL
	};
	
	MY_DEBUG_PRINT(":::::Function Start\n");
	
	/* parse numpy array arguments */
	if (!PyArg_ParseTupleAndKeywords
	    (args, kw, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!dddddd:bin_pixel_quads", kwlist,
	     &PyArray_Type, &in_array_dist1, &PyArray_Type, &in_array_dist2,
	     &PyArray_Type, &in_array_flux1, &PyArray_Type, &in_array_flux2,
	     &PyArray_Type, &in_array_weights1, &PyArray_Type, &in_array_weights2,
	     &PyArray_Type, &in_array_dist3, &PyArray_Type, &in_array_dist4,
	     &PyArray_Type, &in_array_flux3, &PyArray_Type, &in_array_flux4,
	     &PyArray_Type, &in_array_weights3, &PyArray_Type, &in_array_weights4,
	     &PyArray_Type, &in_array_estimator,
	     &PyArray_Type, &out_array,
	     &qso_angle12, &qso_angle34,
	     &x_bin_size, &y_bin_size, &x_bin_count, &y_bin_count))
	{
		return NULL;
	}
	MY_DEBUG_PRINT(":::::After arg parse\n");
	
	bin_pixel_quads_loop(in_array_dist1, in_array_dist2,
			in_array_flux1, in_array_flux2,
			in_array_weights1, in_array_weights2,
			in_array_dist3, in_array_dist4,
			in_array_flux3, in_array_flux4,
			in_array_weights3, in_array_weights4,
			in_array_estimator,
			out_array,
			qso_angle12, qso_angle34,
			x_bin_size, y_bin_size, x_bin_count, y_bin_count);
	
	return NULL;
}

static void pre_allocate_memory(void)
{
	;
}

/* define functions in module */
static PyMethodDef PixelPairMethods[] = {
	{"bin_pixel_pairs", (PyCFunction) bin_pixel_pairs, METH_KEYWORDS,
	 "bin pixel pairs to a 2d array"},
	{"bin_pixel_pairs_histogram", (PyCFunction) bin_pixel_pairs_histogram, METH_KEYWORDS,
	 "create a 3d histogram of 2 displacement axes and flux product"},
	{"bin_pixel_quads", (PyCFunction) bin_pixel_quads, METH_KEYWORDS,
	 "calculate the covariance matrix, given a pre-calculated correlation estimator array"},
	{NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC initbin_pixel_pairs(void)
{
	(void)Py_InitModule("bin_pixel_pairs", PixelPairMethods);
	/* IMPORTANT: this must be called */
	import_array();

	pre_allocate_memory();
}
