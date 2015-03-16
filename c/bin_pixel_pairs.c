/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

/* this is a slightly modified version of: 
 * https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#ifdef _MY_DEBUG_PRINT
#define MY_DEBUG_PRINT(...) PySys_WriteStdout(__VA_ARGS__);
#else
#define MY_DEBUG_PRINT(...)
#endif

static void bin_pixel_pairs_loop(PyArrayObject* in_array_z1, PyArrayObject* in_array_z2, 
				 PyArrayObject* in_array_dist1, PyArrayObject* in_array_dist2, 
				 PyArrayObject* in_array_flux1, PyArrayObject* in_array_flux2, 
				 PyArrayObject* in_array_weights1, PyArrayObject* in_array_weights2, 
				 PyArrayObject* out_array, double qso_angle,
				 double x_bin_size, double y_bin_size,
				 double x_bin_count, double y_bin_count)
{
    int i,j;
    int z1_size, z2_size;
    int bin_x, bin_y;
    int last_z2_start, first_pair_z2;
    double z1, z2, dist1, dist2, flux1, flux2, weight1, weight2;
    double *p_current_bin_flux, *p_current_bin_weight, *p_current_bin_count;
    
    /*  iterate over the arrays */
    z1_size = PyArray_DIM(in_array_z1, 0);
    z2_size = PyArray_DIM(in_array_z2, 0);
   
    MY_DEBUG_PRINT(":::::Before loop\n");

    z1 = 1;
    last_z2_start = 0;
    for (i=0;i<z1_size && z1;i++)
    {
	/* MY_DEBUG_PRINT(":::::Outside iter, i=%d\n", i);*/
	  
	z1 = *((double*) PyArray_GETPTR1(in_array_z1, i));
	dist1 = *((double*) PyArray_GETPTR1(in_array_dist1, i));
	flux1 = *((double*) PyArray_GETPTR1(in_array_flux1, i));
	weight1 = *((double*) PyArray_GETPTR1(in_array_weights1, i));

	z2 = 1;
	/* z values are ordered, so if any z2 was too low to be close enough to the previous z1,
	 * the same should hold for the current z1. */
	first_pair_z2 = 0;
	for(j=last_z2_start;j<z2_size && z2;j++)
	{
	    z2 = *((double*) PyArray_GETPTR1(in_array_z2, j));
	    dist2 = *((double*) PyArray_GETPTR1(in_array_dist2, j));
	    flux2 = *((double*) PyArray_GETPTR1(in_array_flux2, j));
	    weight2 = *((double*) PyArray_GETPTR1(in_array_weights2, j));
	    
	    /* r|| = abs(r1 - r2) */
	    bin_x = abs(dist1 - dist2) / x_bin_size;
	    /* r_ =  (r1 + r2)/2 * qso_angle */
	    bin_y = (dist1 + dist2) * qso_angle / (2. * y_bin_size);

	    if ((bin_x >= 0 && bin_x < x_bin_count) &&
	      (bin_y >= 0 && bin_y < y_bin_count))
	    {
	        /* pixel is in range */
		if (!first_pair_z2)
		    first_pair_z2 = j;
		
		p_current_bin_flux = (double*) PyArray_GETPTR3(out_array, bin_x, bin_y, 0);
		(*p_current_bin_flux) += flux1*flux2;
		p_current_bin_count = (double*) PyArray_GETPTR3(out_array, bin_x, bin_y, 1);
		(*p_current_bin_count) += 1;
		p_current_bin_weight = (double*) PyArray_GETPTR3(out_array, bin_x, bin_y, 2);
		(*p_current_bin_weight) += weight1*weight2;
	    }
	    else
	    {
		/* in flat geometry we cannot move in and out of range more than once. */
		if (first_pair_z2)
		    break;
	    }
	}
	if (first_pair_z2)
	    last_z2_start = first_pair_z2;
    }
}

/*  wrapped function */
static PyObject* bin_pixel_pairs(PyObject* self, PyObject* args, PyObject* kw)
{

    PyArrayObject *in_array_z1;
    PyArrayObject *in_array_z2;
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
    npy_intp out_dim[3] = {0};
        
    static char *kwlist[] = {"ar_z1", "ar_z2", "ar_dist1", "ar_dist2", 
	"ar_flux1", "ar_flux2", "ar_weights1", "ar_weights2",
	"qso_angle", "x_bin_size", "y_bin_size", "x_bin_count", "y_bin_count", NULL};

    MY_DEBUG_PRINT(":::::Function Start\n");
    
    /*  parse numpy array arguments */
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!O!O!O!O!O!O!O!ddddd:bin_pixel_pairs", kwlist, 
	&PyArray_Type, &in_array_z1, &PyArray_Type, &in_array_z2,
	&PyArray_Type, &in_array_dist1, &PyArray_Type, &in_array_dist2,
	&PyArray_Type, &in_array_flux1, &PyArray_Type, &in_array_flux2,
	&PyArray_Type, &in_array_weights1, &PyArray_Type, &in_array_weights2, 
	&qso_angle, &x_bin_size, &y_bin_size, &x_bin_count, &y_bin_count))
    {
        return NULL;
    }

    MY_DEBUG_PRINT(":::::After arg parse\n");

    /*  construct a 3D output array, (x_bin_count) by (y_bin_count) by (flux, counts, weights) */
    out_dim[0] = x_bin_count;
    out_dim[1] = y_bin_count;
    out_dim[2] = 3;
    out_array = (PyArrayObject*) PyArray_ZEROS(3, out_dim, NPY_DOUBLE, 0);
    if (out_array == NULL)
        return NULL;
    
    MY_DEBUG_PRINT(":::::Created out array\n");

    bin_pixel_pairs_loop(in_array_z1, in_array_z2, 
			 in_array_dist1, in_array_dist2, 
			 in_array_flux1, in_array_flux2, 
			 in_array_weights1, in_array_weights2, 
			 out_array, qso_angle, 
			 x_bin_size, y_bin_size, 
			 x_bin_count, y_bin_count
			);

    /*Py_INCREF(out_array);*/
    return (PyObject*) out_array;

    /*  in case bad things happen */
    /*fail:
        Py_XDECREF(out_array);
        return NULL;*/
}

static void pre_allocate_memory(void)
{
    ;
}

/*  define functions in module */
static PyMethodDef PixelPairMethods[] =
{
     {"bin_pixel_pairs", (PyCFunction) bin_pixel_pairs, METH_KEYWORDS,
         "evaluate the cosine on a numpy array"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
initbin_pixel_pairs(void)
{
     (void) Py_InitModule("bin_pixel_pairs", PixelPairMethods);
     /* IMPORTANT: this must be called */
     import_array();
     
     pre_allocate_memory();
     return(0);
}
