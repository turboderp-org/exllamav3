py::class_<BC_GatedRMSNorm, std::shared_ptr<BC_GatedRMSNorm>>(m, "BC_GatedRMSNorm").def
(
    py::init<
        at::Tensor,
        float,
        float,
        int,
        bool
    >(),
    py::arg("weight"),
    py::arg("rms_norm_eps"),
    py::arg("constant_bias"),
    py::arg("w_groups") = 1,
    py::arg("gate_first") = false
)
.def("run", &BC_GatedRMSNorm::run);
