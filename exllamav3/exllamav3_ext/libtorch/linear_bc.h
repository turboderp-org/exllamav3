py::class_<BC_LinearFP16, std::shared_ptr<BC_LinearFP16>>(m, "BC_LinearFP16").def
(
    py::init<
        at::Tensor,
        c10::optional<at::Tensor>
    >(),
    py::arg("weight"),
    py::arg("bias") = py::none()
)
.def("run", &BC_LinearFP16::run);
//.def("run_cublas", &BC_LinearFP16::run_cublas);

py::class_<BC_LinearEXL3, std::shared_ptr<BC_LinearEXL3>>(m, "BC_LinearEXL3").def
(
    py::init<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        int,
        c10::optional<at::Tensor>,
        bool,
        bool,
        at::Tensor
    >(),
    py::arg("trellis"),
    py::arg("suh"),
    py::arg("svh"),
    py::arg("K"),
    py::arg("bias"),
    py::arg("mcg"),
    py::arg("mul1"),
    py::arg("xh")
)
.def("run", &BC_LinearEXL3::run);
