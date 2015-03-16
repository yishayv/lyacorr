/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

/* this is a slightly modified version of: 
 * https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static void bin_pixel_pairs_loop(PyArrayObject* in_array_z1, PyArrayObject* in_array_z2, 
				 PyArrayObject* in_array_dist1, PyArrayObject* in_array_dist2, 
				 PyArrayObject* in_array_flux1, PyArrayObject* in_array_flux2, 
				 PyArrayObject* in_array_weights1, PyArrayObject* in_array_weights2, 
				 PyArrayObject* out_array)
{
    int i,j;
    int z1_size, z2_size;
    int bin_x, bin_y;
    int last_z2_start, first_pair_z2;
    double z1, z2;
    double* p_current_bin;
    
    /*  iterate over the arrays */
    z1_size = PyArray_DIM(in_array_z1, 0);
    z2_size = PyArray_DIM(in_array_z2, 0);
   
    PySys_WriteStdout(":::::Before loop\n");

    z1 = 1;
    last_z2_start = 0;
    for (i=0;i<z1_size && z1;i++)
    {
	/* PySys_WriteStdout(":::::Outside iter, i=%d\n", i);*/
	  
	z1 = *((double*) PyArray_GETPTR1(in_array_z1, i));
	z2 = 1;
	
	/* z values are ordered, so if any z2 was too low to be close enough to the previous z1,
	 * the same should hold for the current z1. */
	first_pair_z2 = 0;
	for(j=last_z2_start;j<z2_size && z2;j++)
	{
	    z2 = *((double*) PyArray_GETPTR1(in_array_z2, j));
	    bin_x = i;
	    bin_y = j;
	    if ((bin_x > 0 && bin_x < 25) &&
	      (bin_y > 0 && bin_y < 25))
	    {
	        /* pixel is in range */
		if (!first_pair_z2)
		    first_pair_z2 = j;
		
		p_current_bin = (double*) PyArray_GETPTR2(out_array, bin_x, bin_y);
		(*p_current_bin) += (double)(z1*z2);
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
    npy_intp out_dim[2] = {25,25};
        
    static char *kwlist[] = {"ar_z1", "ar_z2", "ar_dist1", "ar_dist2", 
	"ar_flux1", "ar_flux2", "ar_weights1", "ar_weights2", NULL};

    PySys_WriteStdout(":::::Function Start\n");
    
    /*  parse numpy array arguments */
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O!O!O!O!O!O!O!O!:bin_pixel_pairs", kwlist, 
	&PyArray_Type, &in_array_z1, &PyArray_Type, &in_array_z2,
	&PyArray_Type, &in_array_dist1, &PyArray_Type, &in_array_dist2,
	&PyArray_Type, &in_array_flux1, &PyArray_Type, &in_array_flux2,
	&PyArray_Type, &in_array_weights1, &PyArray_Type, &in_array_weights2))
    {
        return NULL;
    }

    PySys_WriteStdout(":::::After arg parse\n");

    /*  construct the output array, 25x25 */
    out_array = (PyArrayObject*) PyArray_ZEROS(2, out_dim, NPY_DOUBLE, 0);
    if (out_array == NULL)
        return NULL;
    
    PySys_WriteStdout(":::::Created out array\n");

    bin_pixel_pairs_loop(in_array_z1, in_array_z2, 
			 in_array_dist1, in_array_dist2, 
			 in_array_flux1, in_array_flux2, 
			 in_array_weights1, in_array_weights2, 
			 out_array);

    Py_INCREF(out_array);
    return (PyObject*) out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
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
}
