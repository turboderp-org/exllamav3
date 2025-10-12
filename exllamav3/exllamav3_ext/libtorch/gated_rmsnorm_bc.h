py::class_<BC_GatedRMSNorm, std::shared_ptr<BC_GatedRMSNorm>>(m, "BC_GatedRMSNorm").def
(
    py::init<
        at::Tensor,
        float,
        float
    >(),
    py::arg("weight"),
    py::arg("rms_norm_eps"),
    py::arg("constant_bias")
)
.def("run", &BC_GatedRMSNorm::run);
