/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

/* this is a slightly modified version of: 
 * https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/*  wrapped function */
static PyObject* bin_pixel_pairs(PyObject* self, PyObject* args)
{

    PyArrayObject *in_array1;
    PyArrayObject *in_array2;
    PyObject      *out_array;
    NpyIter *in_iter1;
    NpyIter *in_iter2;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext1;
    NpyIter_IterNextFunc *in_iternext2;
    NpyIter_IterNextFunc *out_iternext;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!O!:bin_pixel_pairs", &PyArray_Type, &in_array1, &PyArray_Type, &in_array2))
        return NULL;

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array1, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    in_iter1 = NpyIter_New(in_array1, NPY_ITER_READONLY, NPY_KEEPORDER,
                             NPY_NO_CASTING, NULL);
    if (in_iter1 == NULL)
        goto fail;
    
    in_iter2 = NpyIter_New(in_array2, NPY_ITER_READONLY, NPY_KEEPORDER,
                             NPY_NO_CASTING, NULL);
    if (in_iter2 == NULL)
        goto fail;

    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        NpyIter_Deallocate(in_iter1);
        goto fail;
    }

    in_iternext1 = NpyIter_GetIterNext(in_iter1, NULL);
    in_iternext2 = NpyIter_GetIterNext(in_iter2, NULL);
    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (in_iternext1 == NULL || in_iternext2 == NULL || out_iternext == NULL) {
        NpyIter_Deallocate(in_iter1);
        NpyIter_Deallocate(in_iter2);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }
    double ** in_dataptr1 = (double **) NpyIter_GetDataPtrArray(in_iter1);
    double ** in_dataptr2 = (double **) NpyIter_GetDataPtrArray(in_iter2);
    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);

    double* p_value5 = (double*) PyArray_GETPTR1(in_array1, 5);
    /* *p_value5 = 5; */
    /*  iterate over the arrays */
    do {
        **out_dataptr = cos(**in_dataptr1)*(**in_dataptr2);
    } while(in_iternext1(in_iter1) && in_iternext2(in_iter2) && out_iternext(out_iter));

    /*  clean up and return the result */
    NpyIter_Deallocate(in_iter1);
    NpyIter_Deallocate(in_iter2);
    NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

static void pre_allocate_memory()
{
    ;
    /* TODO */
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
