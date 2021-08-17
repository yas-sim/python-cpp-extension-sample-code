#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <opencv2/opencv.hpp>

extern "C" {

// Simple addition test function (return = in1 + in2)
static PyObject* test_func(PyObject* self, PyObject* args) {
    int input1, input2;
    if (!PyArg_ParseTuple(args, "ii", &input1, &input2)) {      // Arguments are 2 integer values ("ii")
        return nullptr;
    }
    PyObject *return_value = PyLong_FromLong(input1 + input2);  // Create object to return
    return return_value;
}

// Simple image processing function (Inverse image color)
static PyObject* test_image_processing(PyObject* self, PyObject* args) {
    PyArrayObject* input_image;
    PyObject* output_image;
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_image)) {            // Argument is an Numpy array object
        return nullptr;
    }
    output_image = PyArray_NewLikeArray(input_image, NPY_ANYORDER, NULL, 0);    // Create object to return

    int ndim = PyArray_NDIM(input_image);                           // Number of dimensions of the input Numpy array
    npy_intp *shape;
    shape = PyArray_SHAPE(input_image);                             // Shape of the input Numpy array (npy_intp[])

    // Obtain data buffer pointers
    char* in_buf  = static_cast<char*>(PyArray_DATA(input_image));   // PyArray_DATA() will return void*
    char* out_buf = static_cast<char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output_image)));

    // Image processing
    size_t C = shape[2];
    size_t W = shape[1];
    size_t H = shape[0];
    for(size_t y=0; y<H; y++) {
        for(size_t x=0; x<W; x++){
            for(size_t c=0; c<C; c++) {
                out_buf[y*W*C + x*C + c] = 255 - in_buf[y*W*C + x*C + c];
            }
        }
    }

    return output_image;    // Returned object must exist after exiting of this C++ function
}

// Simple image processing function using OpenCV (Canny edge detection)
static PyObject* test_image_processing_OCV(PyObject* self, PyObject* args) {
    PyArrayObject* input_image;
    int th1, th2;               // Threshold value for Canny edge detection
    if(!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &input_image, &th1, &th2)) {  // 3 arguments (Numpy obj, int, int)
        return nullptr;
    }

    npy_intp* shape = PyArray_SHAPE(input_image);

    char* in_buf = static_cast<char*>(PyArray_DATA(input_image));          // PyArray_DATA() will return void*

    // Image processing with OpenCV
    cv::Mat in_mat(shape[0] /*Width(rows)*/, shape[1] /*Height(cols)*/, CV_8UC3, in_buf);   // Input cv::Mat generated from input Numpy object
    cv::Mat out_mat;                                    // Mat object to store final image
    cv::Mat channels[3];                                // Mat object for channel extraction
    cv::cvtColor(in_mat, in_mat, cv::COLOR_BGR2GRAY);   // Color -> Gray scale
    cv::split(in_mat, channels);                        // Channel extraction. packed BGR -> [B, G, R]
    cv::Canny(channels[0], out_mat, th1, th2);          // Canny edge detection

    npy_intp out_shape[] = { out_mat.rows, out_mat.cols, out_mat.channels() };
    PyObject* result = PyArray_SimpleNew(3, out_shape, NPY_UINT8);                              // Create a Numpy object to return
    char* out_buf = static_cast<char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result))); // Obtain data buffer pointer of the final image
    memcpy(out_buf, out_mat.data, out_shape[0] * out_shape[1] * out_shape[2]);                  // Copy data buffer

    return result;          // Return created Numpy object
}

// Numpy array attribute extraction test function
static PyObject* npy_array_test(PyObject* self, PyObject* args) {
    PyArrayObject* input_image;

    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_image)) {        // Argument is a Numpy object
        return nullptr;
    }

    size_t ndims = PyArray_NDIM(input_image);                               // Number of dimensions
    std::cout << "#Dims: " << PyArray_NDIM(input_image) << std::endl;

    npy_intp* shape = PyArray_SHAPE(input_image);                           // Shape
    for(size_t i=0; i<ndims; i++) {
        std::cout << "Dim" << i << ": " << shape[i] << std::endl;
    }
    std::cout << "NpyType: " << PyArray_TYPE(input_image) << std::endl;         // Numpy data type
    std::cout << "ItemSize: " << PyArray_ITEMSIZE(input_image) << std::endl;    // Size of item (element)
    std::cout << "TotalSize: " << PyArray_SIZE(input_image) << std::endl;       // Total size (in size of item)
    std::cout << "TotalBytes: " << PyArray_NBYTES(input_image) << std::endl;    // Total bytes
    uint8_t* buffer = static_cast<uint8_t*>(PyArray_DATA(input_image));         // PyArray_DATA() will return void*

    // Show first 10 items of the Numpy array
    int bytes = PyArray_NBYTES(input_image);
    int disp_size = bytes<10 ? bytes : 10;
    for(size_t i=0; i<disp_size; i++) {
        std::cout << (int)buffer[i] << " ";
    }
    std::cout << std::endl;

    return PyLong_FromLong(0);
}

// Function definition table to export to Python
PyMethodDef method_table[] = {
    {"test_func", static_cast<PyCFunction>(test_func), METH_VARARGS, "test method function of test module"},
    {"test_image_processing", static_cast<PyCFunction>(test_image_processing), METH_VARARGS, "test image processing function"},
    {"test_image_processing_OCV", static_cast<PyCFunction>(test_image_processing_OCV), METH_VARARGS, "test image processing function"},
    {"npy_array_test", static_cast<PyCFunction>(npy_array_test), METH_VARARGS, "test numpy array handling test"},
    {NULL, NULL, 0, NULL}
};

// Module definition table
PyModuleDef test_module = {
    PyModuleDef_HEAD_INIT,
    "python_cpp_module",
    "test module",
    -1,
    method_table
};

// Initialize and register module function
// Function name must be 'PyInit_'+module name
// This function must be the only *non-static* function in the source code
PyMODINIT_FUNC PyInit_python_cpp_module(void) {
    import_array();                                 // Required to receive Numpy object as arguments
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return PyModule_Create(&test_module);
}

} // extern "C"