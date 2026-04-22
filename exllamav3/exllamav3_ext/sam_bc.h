py::class_<BC_SAM, std::shared_ptr<BC_SAM>>(m, "BC_SAM").def(py::init<>())
.def("reset", &BC_SAM::reset)
.def("accept", &BC_SAM::accept)
.def("accept_tensor", &BC_SAM::accept_tensor)
.def("length", &BC_SAM::length);
