/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

/* this is a slightly modified version of: 
 * https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/*  wrapped function */
static PyObject* bin_pixel_pairs(PyObject* self, PyObject* args)
{

    PyArrayObject *in_array1;
    PyArrayObject *in_array2;
    PyArrayObject *out_array;
    double z1, z2;
    int z1_size, z2_size;
    int i,j;
    int bin_x, bin_y;
    npy_intp out_dim[2] = {25,25};
    double* p_current_bin;

    /*  parse numpy array arguments */
    if (!PyArg_ParseTuple(args, "O!O!:bin_pixel_pairs", &PyArray_Type, &in_array1, &PyArray_Type, &in_array2))
        return NULL;

    /*  construct the output array, 25x25 */
    out_array = (PyArrayObject*) PyArray_SimpleNew(2, out_dim, NPY_DOUBLE);
    if (out_array == NULL)
        return NULL;

    /* double* p_value5 = (double*) PyArray_GETPTR1(in_array1, 5); */
    /* *p_value5 = 5; */
    /*  iterate over the arrays */
    z1_size = PyArray_DIM(in_array1, 0);
    z2_size = PyArray_DIM(in_array2, 0);
   
    for (i=0;i<z1_size;i++)
    {
	z1 = *((double*) PyArray_GETPTR1(in_array1, i));
	for(j=0;j<z2_size;j++)
	{
	    z2 = *((double*) PyArray_GETPTR1(in_array2, j));
	    bin_x = z1;
	    bin_y = z2;
	    if ((bin_x > 0 && bin_x < 25) &&
	      (bin_y > 0 && bin_y < 25))
	    {
		p_current_bin = (double*) PyArray_GETPTR2(out_array, i, j);
		(*p_current_bin) = (double)(z1*z2);
	    }
	}
    }

    Py_INCREF(out_array);
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
     {"bin_pixel_pairs", bin_pixel_pairs, METH_VARARGS,
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
