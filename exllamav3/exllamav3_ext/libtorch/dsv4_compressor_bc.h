py::class_<BC_DSV4Compressor, std::shared_ptr<BC_DSV4Compressor>>(m, "BC_DSV4Compressor").def
(
    py::init<
        std::shared_ptr<BC_LinearEXL3>,
        std::shared_ptr<BC_LinearFP16>,
        std::shared_ptr<BC_LinearEXL3>,
        std::shared_ptr<BC_LinearFP16>,
        at::Tensor,
        at::Tensor,
        float,
        at::Tensor,
        int,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        c10::optional<at::Tensor>,
        c10::optional<at::Tensor>,
        c10::optional<at::Tensor>,
        c10::optional<at::Tensor>
    >(),
    py::arg("wkv_exl3"),
    py::arg("wkv_fp16"),
    py::arg("wgate_exl3"),
    py::arg("wgate_fp16"),
    py::arg("ape"),
    py::arg("norm_w"),
    py::arg("rms_norm_eps"),
    py::arg("inv_freq"),
    py::arg("m"),
    py::arg("kv_scratch"),
    py::arg("gate_scratch"),
    py::arg("xh_scratch"),
    py::arg("mg_trellis") = c10::nullopt,
    py::arg("mg_suh") = c10::nullopt,
    py::arg("mg_svh") = c10::nullopt,
    py::arg("mg_indices") = c10::nullopt
)
.def("run", &BC_DSV4Compressor::run);
