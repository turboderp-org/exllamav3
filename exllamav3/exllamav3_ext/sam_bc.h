py::class_<BC_SAM, std::shared_ptr<BC_SAM>>(m, "BC_SAM").def(py::init<>())
.def("reset", &BC_SAM::reset)
.def("accept", &BC_SAM::accept)
.def("accept_tensor", &BC_SAM::accept_tensor)
.def("export_csr", &BC_SAM::export_csr)
.def("length", &BC_SAM::length);

py::class_<FrozenSAMCursor>(m, "FrozenSAMCursor")
.def(py::init<std::vector<at::Tensor>>())
.def("draft", &FrozenSAMCursor::draft);
