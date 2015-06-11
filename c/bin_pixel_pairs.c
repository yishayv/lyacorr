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

static void
bin_pixel_pairs_loop(PyArrayObject * in_array_dist1,
					 PyArrayObject * in_array_dist2,
					 PyArrayObject * in_array_flux1,
					 PyArrayObject * in_array_flux2,
					 PyArrayObject * in_array_weights1,
					 PyArrayObject * in_array_weights2,
					 PyArrayObject * out_array, double qso_angle,
					 double x_bin_size, double y_bin_size,
					 double x_bin_count, double y_bin_count)
{
	int i, j;
	int dist1_size, dist2_size;
	int bin_x, bin_y;
	int last_dist2_start, first_pair_dist2;
	int max_dist2_index;
	double dist1, dist2, flux1, flux2, weight1, weight2;
	double *p_current_bin_flux, *p_current_bin_weight, *p_current_bin_count;
	double weighted_flux1, weighted_flux2;
	double x_scale, y_scale;
	double max_dist_for_qso_angle;

	/* iterate over the arrays */
	dist1_size = PyArray_DIM(in_array_dist1, 0);
	dist2_size = PyArray_DIM(in_array_dist2, 0);
	
	if (dist1_size && dist2_size)
	{
		/* if dist1 is nearer, it is more efficient to run the function with 1 and 2 reversed. */
		dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, 0));
		dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, 0));
		if (dist1 < dist2)
		{
			bin_pixel_pairs_loop(in_array_dist2,
					 in_array_dist1,
					 in_array_flux2,
					 in_array_flux1,
					 in_array_weights2,
					 in_array_weights1,
					 out_array, qso_angle,
					 x_bin_size, y_bin_size,
					 x_bin_count, y_bin_count);
			return;
		}
	}
	else
	{
		return;
	}

	x_scale = 1. / x_bin_size;
	y_scale = qso_angle / (2. * y_bin_size);
	max_dist_for_qso_angle = y_bin_count / y_scale;

	/*
	 * find the largest index of dist2 for which a transverse distance to the other
	 * QSO is within range.
	 */
	dist2 = 1;
	/* set initial index to the end of the array. */
	max_dist2_index = dist2_size;
	for (j = 0; j < dist2_size && dist2; j++)
	{
		dist2 = *((double *)PyArray_GETPTR1(in_array_dist2, j));
		if (dist2 > max_dist_for_qso_angle)
		{
			max_dist2_index = j;
			break;
		}
	}
	
	MY_DEBUG_PRINT(":::::Before loop\n");

	dist1 = 1;
	last_dist2_start = 0;
	for (i = 0; i < dist1_size && dist1; i++)
	{
		/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i); */
		dist1 = *((double *)PyArray_GETPTR1(in_array_dist1, i));
		flux1 = *((double *)PyArray_GETPTR1(in_array_flux1, i));
		weight1 = *((double *)PyArray_GETPTR1(in_array_weights1, i));

		if (dist1 > max_dist_for_qso_angle)
			break;

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
			
			/* r|| = abs(r1 - r2) */
			bin_x = (int)(fabs(dist1 - dist2) * x_scale);
			/* r_ = (r1 + r2)/2 * qso_angle */
			bin_y = (int)((dist1 + dist2) * y_scale);
			
			weighted_flux2 = flux2 * weight2;

			if ((bin_x < x_bin_count) && (bin_y < y_bin_count))
			{
				/* pixel is in range */
				if (!first_pair_dist2)
					first_pair_dist2 = j;

				p_current_bin_flux = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, 0);
				weighted_flux1 = flux1 * weight1;
				(*p_current_bin_flux) += weighted_flux1 * weighted_flux2;
				p_current_bin_count = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, 1);
				(*p_current_bin_count) += 1;
				p_current_bin_weight = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, 2);
				(*p_current_bin_weight) += weight1 * weight2;
			}
			else
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
static PyObject *bin_pixel_pairs(PyObject * self, PyObject * args,
								 PyObject * kw)
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
						 out_array, qso_angle,
						 x_bin_size, y_bin_size, x_bin_count, y_bin_count);

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
					 double f_min, double f_max,
					 double f_bin_count, double* p_pair_count)
{
	int i, j;
	int dist1_size, dist2_size;
	int bin_x, bin_y, bin_f;
	int last_dist2_start, first_pair_dist2;
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
					bin_f = f_bin_count-1;
				else if (flux_product < f_min)
					bin_f = 0;
				else
					bin_f = (int)(f_bin_count*(flux_product - f_min)/(f_max - f_min));

				p_current_bin_flux = (double *)PyArray_GETPTR3(out_array, bin_x, bin_y, bin_f);
				/* MY_DEBUG_PRINT(":::::Adding pixel, %d, %d, %d, weight:%lf\n", 
					bin_x, bin_y, bin_f, weight1*weight2); */
				(*p_current_bin_flux) += weight1 * weight2;
				(*p_pair_count)++;
			}
			else
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
	PyObject* ret_val;

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
		(args, kw, "O!O!O!O!O!O!O!ddddddddd:bin_pixel_pairs", kwlist,
		 &PyArray_Type, &in_array_dist1, &PyArray_Type, &in_array_dist2,
		 &PyArray_Type, &in_array_flux1, &PyArray_Type, &in_array_flux2,
		 &PyArray_Type, &in_array_weights1, &PyArray_Type, &in_array_weights2,
		 &PyArray_Type, &out_array,
		 &qso_angle, &x_bin_size, &y_bin_size, &x_bin_count, &y_bin_count,
		 &f_min, &f_max, &f_bin_count, &pair_count))
	{
		return NULL;
	}
	MY_DEBUG_PRINT(":::::After arg parse\n");

	bin_pixel_pairs_histogram_loop(in_array_dist1, in_array_dist2,
						 in_array_flux1, in_array_flux2,
						 in_array_weights1, in_array_weights2,
						 out_array, qso_angle,
						 x_bin_size, y_bin_size, x_bin_count, y_bin_count,
						 f_min, f_max, f_bin_count, &pair_count);

	ret_val = PyFloat_FromDouble(pair_count);
	return ret_val;
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
	{NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC initbin_pixel_pairs(void)
{
	(void)Py_InitModule("bin_pixel_pairs", PixelPairMethods);
	/* IMPORTANT: this must be called */
	import_array();

	pre_allocate_memory();
	return;
}
