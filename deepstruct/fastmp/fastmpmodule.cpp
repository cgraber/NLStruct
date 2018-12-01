#include <Python.h>
#include <iostream>
#include <random>
#include <algorithm>
#include "Region.h"
#include <numpy/arrayobject.h>
#include "structmember.h"
#include <thread>
#include <chrono>
#include <omp.h>


/*************************
 *FastMP class definition*
 *************************
 */
typedef struct {
    PyObject_HEAD
    int num_vals;
    int num_nodes;
    int num_pair_regions;
    int belief_size;
    float* potentials;
    float* beliefs;
    PyObject *beliefs_obj;
    //float* belief_wts;
    PyObject *potentials_obj;
    PyObject *msgs_obj;
    float *lambda_msgs;
    //PyObject *belief_sum_wts_obj;
    MPGraph<float, int> *g;
} FastMP;

static void FastMP_dealloc(FastMP* self)
{
    PyArray_Free((PyObject*)self->potentials_obj, self->potentials);
    if (self->beliefs != NULL) 
    {
        PyArray_Free((PyObject*)self->beliefs_obj, self->beliefs);
        Py_DECREF(self->beliefs_obj);
    }
    Py_DECREF(self->potentials_obj);
    delete self->g;
    

    //TODO: Do we need to free anything within the graph? Potentials?
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int FastMP_init(FastMP *self, PyObject *args, PyObject *kwds)
{
    PyObject *pair_regions;
    if (!PyArg_ParseTuple(args, "iiOO", &self->num_nodes, &self->num_vals,
                &pair_regions, &(self->potentials_obj)))
    {
        return -1;
    }
    Py_INCREF(self->potentials_obj); //Since we hold onto this bad boy
    npy_intp* pot_obj_dims = PyArray_DIMS(self->potentials_obj);
    
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->potentials_obj));

    if (PyArray_AsCArray(&(self->potentials_obj), (void *)& self->potentials, pot_obj_dims, 1, descr) < 0) {
        Py_DECREF(self->potentials_obj);
        PyErr_SetString(PyExc_TypeError, "error converting node potentials to c array");
        return -1;
    }
    self->g = new MPGraph<float, int>();
    //Build graph info
    std::vector<int> cards(self->num_nodes, self->num_vals);
    self->g->AddVariables(cards);
    std::vector<MPGraph<float, int>::PotentialID> pots;
    std::vector<MPGraph<float, int>::RegionID> regs;
    //First, register node potentials
    for (int pot_i = 0, node_i = 0; pot_i < self->num_nodes*self->num_vals; node_i++, pot_i = pot_i + self->num_vals) 
    {
        pots.push_back(self->g->AddPotential(MPGraph<float, int>::PotentialVector(&(self->potentials[pot_i]), self->num_vals)));
    }
        
    int region_count = self->num_nodes;
    if (PyList_Check(pair_regions)) {
        self->num_pair_regions = PyList_Size(pair_regions);
        const size_t increment_val = self->num_vals * self->num_vals;

        //NOTE: We have to add *all* of the potentials before we add regions, or else segfaults happen
        for (Py_ssize_t pair_ind = 0, pot_i = self->num_nodes*self->num_vals; pair_ind < self->num_pair_regions; pair_ind++, pot_i += increment_val) {
            pots.push_back(self->g->AddPotential(MPGraph<float, int>::PotentialVector(&(self->potentials[pot_i]), increment_val)));
        }
        //Add node regions
        for (int pot_i = 0, node_i = 0; pot_i < self->num_nodes*self->num_vals; node_i++, pot_i = pot_i + self->num_vals) 
        {
            regs.push_back(self->g->AddRegion(1.0, std::vector<int>{node_i}, pots[node_i]));
        }

        //Add pair regions
        for (Py_ssize_t pair_ind = 0, pot_i = self->num_nodes*self->num_vals; pair_ind < self->num_pair_regions; pair_ind++, pot_i += increment_val) {
            PyObject *pair = PyList_GetItem(pair_regions, pair_ind);
            if (PyTuple_Check(pair)) {
                int first_node = PyLong_AsLong(PyTuple_GetItem(pair, (Py_ssize_t) 0));
                int second_node = PyLong_AsLong(PyTuple_GetItem(pair, (Py_ssize_t) 1));
                regs.push_back(self->g->AddRegion(1.0, {first_node, second_node}, pots[region_count]));
                self->g->AddConnection(regs[first_node], regs[region_count]);
                self->g->AddConnection(regs[second_node], regs[region_count++]);
            } else {
                PyErr_SetString(PyExc_TypeError, "pairs are not tuples!");
                return -1;
            }
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "pair_regions not a list!");
        return -1;
    }
    self->g->AllocateMessageMemory();

    self->belief_size = self->g->ComputeBeliefSize(false);
    self->beliefs = NULL;
    return 0;

}

static PyObject* fastmp_get_num_msgs(FastMP *self, PyObject *args)
{
    return Py_BuildValue("i", self->g->GetLambdaSize());
}

static PyObject* fastmp_get_num_beliefs(FastMP *self, PyObject *args)
{
    return Py_BuildValue("i", self->belief_size);
}

static PyObject* fastmp_runmp(FastMP *self, PyObject *args)
{
    int num_iters;
    float eps;
    if (!PyArg_ParseTuple(args, "if", &num_iters, &eps))
    {
        return NULL;
    }
    RMP<float, int> RMPAlgo(*self->g);
    float obj_val = RMPAlgo.RunMP(self->lambda_msgs, eps, num_iters);
    return Py_BuildValue("f", obj_val);    
}

static PyObject* fastmp_update_potentials(FastMP *self, PyObject *args)
{
    PyArray_Free(self->potentials_obj, self->potentials);
    Py_DECREF(self->potentials_obj);
    if (!PyArg_ParseTuple(args, "O", &(self->potentials_obj)))
    {
        return NULL;
    }
    Py_INCREF(self->potentials_obj); 

    npy_intp* pot_obj_dims = PyArray_DIMS(self->potentials_obj);
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->potentials_obj));
    if (PyArray_AsCArray(&(self->potentials_obj), (void *)&self->potentials, pot_obj_dims, 1, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting node potentials to c array");
        return NULL;
    }

    for (int i = 0; i < self->num_nodes; i++) 
    {
        int pot_ind = i*self->num_vals;
        self->g->ReplacePotential(MPGraph<float, int>::PotentialVector(&(self->potentials[pot_ind]),self->num_vals), i);
    }
    int offset = self->num_nodes * self->num_vals;
    for (int i = self->num_nodes; i < self->num_nodes + self->num_pair_regions; i++) {
        self->g->ReplacePotential(MPGraph<float, int>::PotentialVector(&(self->potentials[offset]), self->num_vals*self->num_vals), i);
        offset += self->num_vals*self->num_vals;
    }
    //PyDimMem_FREE(node_pot_obj_dims);
    //PyDimMem_FREE(pair_pot_obj_dims);

    Py_RETURN_NONE;
}

static PyObject* fastmp_update_msgs(FastMP *self, PyObject *args)
{
    if (self->lambda_msgs != NULL) {
        PyArray_Free(self->msgs_obj, self->lambda_msgs);
        Py_DECREF(self->msgs_obj);
    }
    if (!PyArg_ParseTuple(args, "O", &self->msgs_obj)) {
        return NULL;
    }
    Py_INCREF(self->msgs_obj);

    npy_intp* msg_obj_dims = PyArray_DIMS(self->msgs_obj);
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->msgs_obj));
    if (PyArray_AsCArray(&self->msgs_obj, (void *)&self->lambda_msgs, msg_obj_dims, 1, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting messages to c array");
        return NULL;
    }
    Py_RETURN_NONE;

}

static PyObject* fastmp_update_beliefs_pointer(FastMP *self, PyObject *args)
{
    if (self->beliefs != NULL) {
        PyArray_Free(self->beliefs_obj, self->beliefs);
        Py_DECREF(self->beliefs_obj);
    }
    if (!PyArg_ParseTuple(args, "O", &self->beliefs_obj)) {
        return NULL;
    }
    Py_INCREF(self->beliefs_obj);

    npy_intp* belief_obj_dims = PyArray_DIMS(self->beliefs_obj);
    PyArray_Descr *descr = PyArray_DescrFromType(PyArray_TYPE(self->beliefs_obj));
    if (PyArray_AsCArray(&self->beliefs_obj, (void *)&self->beliefs, belief_obj_dims, 1, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting beliefs to c array");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* fastmp_update_beliefs(FastMP *self, PyObject *args)
{
    float eps;
    if (!PyArg_ParseTuple(args, "f", &eps)) {
        return NULL;
    }
    self->g->ComputeBeliefs(&self->lambda_msgs[0], eps, self->beliefs, false, self->num_vals*self->num_vals);
    Py_RETURN_NONE;
}

static PyObject* fastmp_get_belief(FastMP* self, PyObject* args)
{
    int region_ind, val_ind;
    if (!PyArg_ParseTuple(args, "ii", &region_ind, &val_ind)) return NULL;
    float result = self->g->getBelief(region_ind, val_ind);
    return Py_BuildValue("f", result);
}

static PyObject* fastmp_get_beliefs(FastMP* self, PyObject* args)
{
    return Py_BuildValue("O", self->beliefs_obj);
}

static PyObject* fastmp_get_entropy(FastMP* self, PyObject* args) 
{
    float result = self->g->entropy;
    return Py_BuildValue("f", result);
}

static PyMethodDef FastMP_methods[] = {
    {"get_num_msgs", (PyCFunction)fastmp_get_num_msgs, METH_VARARGS,
     "get number of messages required for message passing"},
    {"get_num_beliefs", (PyCFunction)fastmp_get_num_beliefs, METH_VARARGS,
     "get size of beliefs vector"},
    {"runmp", (PyCFunction)fastmp_runmp, METH_VARARGS,
     "run message passing"},
    {"update_msgs", (PyCFunction)fastmp_update_msgs, METH_VARARGS,
     "update messages"},
    {"update_beliefs_pointer", (PyCFunction)fastmp_update_beliefs_pointer, METH_VARARGS,
     "update beliefs pointer"},
    {"update_beliefs", (PyCFunction)fastmp_update_beliefs, METH_VARARGS,
     "update beliefs"},
    {"get_belief", (PyCFunction)fastmp_get_belief, METH_VARARGS,
     "get belief"},
    {"get_beliefs", (PyCFunction)fastmp_get_beliefs, METH_VARARGS,
     "get all beliefs"},
    {"update_potentials", (PyCFunction)fastmp_update_potentials, METH_VARARGS,
     "update potentials"},
    {"get_entropy", (PyCFunction)fastmp_get_entropy, METH_VARARGS,
                "Get Entropy"},
    {NULL} /* Sentinel */
};

static PyTypeObject FastMPType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fastmp.FastMP",            /* tp_name */
    sizeof(FastMP),             /* tp_basicsize */
    0,                          /* tp_itemsize */
    (destructor)FastMP_dealloc, /* tp_dealloc */
    0,                          /* tp_print */
    0,                          /* tp_getattr */
    0,                          /* tp_setattr */
    0,                          /* tp_compare */
    0,                          /* tp_repr */
    0,                          /* tp_as_number */
    0,                          /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                          /* tp_hash */
    0,                          /* tp_call */
    0,                          /* tp_str */
    0,                          /* tp_getattro */
    0,                          /* tp_setattro */
    0,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,         /* tp_flags */
    "FastMP objects",           /* tp_doc */
    0,                          /* tp_traverse */
    0,                          /* tp_clear */
    0,                          /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    0,                          /* tp_iter */
    0,                          /* tp_iternext */
    FastMP_methods,             /* tp_methods */
    0,                          /* tp_members */
    0,                          /* tp_getset */
    0,                          /* tp_base */
    0,                          /* tp_dict */
    0,                          /* tp_descr_get */
    0,                          /* tp_descr_set */
    0,                          /* tp_dictoffset */
    (initproc)FastMP_init,      /* tp_init */
    0,                          /* tp_alloc */
    PyType_GenericNew,          /* tp_new */
};

/*
 **********************
 *Other module methods*
 **********************
 */

void run_mp(MPGraph<float, int> *mp_graph, float *lambda_msgs, int num_iters, float eps) {
    RMP<float, int> RMPAlgo(*mp_graph);
    float obj_val = RMPAlgo.RunMP(lambda_msgs, eps, num_iters);
}

static PyObject* fastmpmod_runmp(FastMP* self, PyObject* args)
{
    PyObject *mp_graphs;
    int num_iters;
    float eps;
    if (!PyArg_ParseTuple(args, "Oif", &mp_graphs, &num_iters, &eps))
    {
        return NULL;
    }
    if (!PyList_Check(mp_graphs)) {
        PyErr_SetString(PyExc_TypeError, "mp_graphs not a list!");
        return NULL;
    }
    Py_ssize_t num_graphs = PyList_Size(mp_graphs);
    //std::vector<std::thread*> threads;
#pragma omp parallel for
    for (Py_ssize_t graph_ind = 0; graph_ind < num_graphs; graph_ind++) {
        PyObject *graph = PyList_GetItem(mp_graphs, graph_ind);
        MPGraph<float, int> *mp_graph = ((FastMP *)graph)->g;
        float *lambda_msgs = ((FastMP *)graph)->lambda_msgs;
        //threads.push_back(new std::thread(run_mp, mp_graph, lambda_msgs, num_iters, eps));
        run_mp(mp_graph, lambda_msgs, num_iters, eps);
    }
    /*
    for (auto& th: threads) {
        th->join();
        delete th;
    }
    */
    Py_RETURN_NONE;
} 

void update_beliefs(MPGraph<float, int> *mp_graph, float *lambda_msgs, float *beliefs, int num_pair_vals, float eps) {
    
    mp_graph->ComputeBeliefs(lambda_msgs, eps, beliefs, false, num_pair_vals);
}

static PyObject* fastmpmod_update_beliefs(FastMP* self, PyObject* args)
{
    PyObject *mp_graphs;
    float eps;
    if (!PyArg_ParseTuple(args, "Of", &mp_graphs, &eps)) {
        PyErr_SetString(PyExc_TypeError, "mp_graphs not a list!");
        return NULL;
    }
    Py_ssize_t num_graphs = PyList_Size(mp_graphs);
    //std::vector<std::thread*> threads;
    #pragma omp parallel for
    for (Py_ssize_t graph_ind = 0; graph_ind < num_graphs; graph_ind++) {
        PyObject *graph = PyList_GetItem(mp_graphs, graph_ind);
        MPGraph<float, int> *mp_graph = ((FastMP *)graph)->g;
        float *lambda_msgs = ((FastMP *)graph)->lambda_msgs;
        float *beliefs = ((FastMP *)graph)->beliefs;
        int num_pair_vals = ((FastMP *)graph)->num_vals;
        num_pair_vals *= num_pair_vals;
        //threads.push_back(new std::thread(update_beliefs, mp_graph, lambda_msgs, beliefs, num_pair_vals, eps));
        update_beliefs(mp_graph, lambda_msgs, beliefs, num_pair_vals, eps);
    }
    /*
    for (auto& th: threads) {
        th->join();
        delete th;
    }
    */
    Py_RETURN_NONE;
}
struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static PyMethodDef FastMP_module_methods[] = {
    {"runmp", (PyCFunction)fastmpmod_runmp, METH_VARARGS,
     "run multithreaded message passing"},
    {"update_beliefs", (PyCFunction)fastmpmod_update_beliefs, METH_VARARGS,
     "run multithreaded belief update"},
    {NULL, NULL, 0, NULL} //Sentinel
};

static int fastmp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int fastmp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "fastmp", /* name of module */
        NULL,     /* module documentation */
        sizeof(struct module_state), /* size of per-interpreter state of the module */
        FastMP_module_methods,
        NULL,
        fastmp_traverse,
        fastmp_clear,
        NULL
};
/*
 *******************
 *Module init stuff*
 *******************
 */

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC PyInit_fastmp(void)
{
    PyObject *m;
    FastMPType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&FastMPType) < 0)
    {
        return NULL;
    }

    //m = Py_InitModule3("fastmp", FastMP_module_methods,
    //        "Module running fast message passing");
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    Py_INCREF(&FastMPType);
    PyModule_AddObject(m, "FastMP", (PyObject *)&FastMPType);


    import_array(); /* So we can use Numpy Stuff */
    return m;
}

