py::class_<BC_DSV4Attention, std::shared_ptr<BC_DSV4Attention>>(m, "BC_DSV4Attention").def
(
    py::init<
        std::shared_ptr<BC_LinearEXL3>, std::shared_ptr<BC_LinearEXL3>,
        std::shared_ptr<BC_LinearEXL3>, std::shared_ptr<BC_LinearEXL3>,
        std::shared_ptr<BC_LinearEXL3>, c10::optional<at::Tensor>,
        std::shared_ptr<BC_DSV4Compressor>, std::shared_ptr<BC_DSV4Compressor>,
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, bool, bool,
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
        at::Tensor,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>,
        at::Tensor, at::Tensor, at::Tensor,
        int, int, int, int, int, int, int, int, int, int, int, int, int, float, float, int, int,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>
    >()
)
.def("needs_configure", &BC_DSV4Attention::needs_configure)
.def("configure_slot", &BC_DSV4Attention::configure_slot)
.def("run", &BC_DSV4Attention::run);

py::class_<BC_DSV4BatchAttention, std::shared_ptr<BC_DSV4BatchAttention>>(m, "BC_DSV4BatchAttention").def
(
    py::init<
        std::shared_ptr<BC_LinearEXL3>, std::shared_ptr<BC_LinearEXL3>,
        std::shared_ptr<BC_LinearEXL3>, c10::optional<at::Tensor>,
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, bool, bool,
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, bool, bool,
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, float,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, float,
        at::Tensor,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>,
        at::Tensor, at::Tensor,
        int, int, int, int, int, int, int, int, int, int, int, int, int, float, float, int, int,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
        c10::optional<at::Tensor>, c10::optional<at::Tensor>, int, bool, bool
    >()
)
.def("needs_configure", &BC_DSV4BatchAttention::needs_configure)
.def("configure_slot", &BC_DSV4BatchAttention::configure_slot)
.def("run", &BC_DSV4BatchAttention::run);
