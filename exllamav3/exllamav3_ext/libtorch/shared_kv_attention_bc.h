py::class_<BC_SharedKVAttention, std::shared_ptr<BC_SharedKVAttention>>(m, "BC_SharedKVAttention").def
(
    py::init<
        int, int, int, int, int, int,
        std::shared_ptr<BC_LinearEXL3>,
        std::shared_ptr<BC_LinearEXL3>,
        c10::optional<at::Tensor>,
        float, float,
        at::Tensor,
        int, float, float, int, int,
        at::Tensor, at::Tensor, at::Tensor, at::Tensor
    >(),
    py::arg("num_q_heads"),
    py::arg("num_kv_heads"),
    py::arg("head_dim"),
    py::arg("hidden_size"),
    py::arg("hidden_size_padded"),
    py::arg("page_size"),
    py::arg("q_proj"),
    py::arg("o_proj"),
    py::arg("q_norm"),
    py::arg("norm_eps"),
    py::arg("norm_constant_bias"),
    py::arg("inv_freq"),
    py::arg("rope_style"),
    py::arg("attn_factor"),
    py::arg("l4_scaling_beta"),
    py::arg("l4_scaling_original"),
    py::arg("rotate_dims"),
    py::arg("cache_k"),
    py::arg("cache_v"),
    py::arg("xh"),
    py::arg("h32")
)
.def("needs_configure", &BC_SharedKVAttention::needs_configure)
.def("configure_slot", &BC_SharedKVAttention::configure_slot,
    py::arg("bsz"),
    py::arg("q_len"),
    py::arg("q"),
    py::arg("o"),
    py::arg("partial_o"),
    py::arg("partial_ml"),
    py::arg("k_split"),
    py::arg("k_combine"),
    py::arg("block_n"),
    py::arg("splits_cap"),
    py::arg("programs"),
    py::arg("xp"),
    py::arg("yp")
)
.def("run", &BC_SharedKVAttention::run);
