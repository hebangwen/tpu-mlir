# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class Gemma4Converter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.do_vit = True
        self.vit_f16_out_bf16 = True
        self.rmsnorm_type = WeightType.ZEROCENTERED_RMSNORM

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = GEMMA4_INFO
        self.llm_config = config.text_config
        self.llm_type = LlmType.GEMMA4

        # Layer type mapping and per-layer head dim
        self.layer_types = self.llm_config.layer_types
        global_head_dim = getattr(self.llm_config, 'global_head_dim', None)
        self._gemma4_head_dim_override = {}
        for i, lt in enumerate(self.layer_types):
            if lt == "full_attention" and global_head_dim:
                self._gemma4_head_dim_override[i] = global_head_dim
            else:
                self._gemma4_head_dim_override[i] = self.llm_config.head_dim

        # KV sharing: last num_kv_shared_layers share K/V from earlier matching layers
        num_kv_shared = getattr(self.llm_config, 'num_kv_shared_layers', 0)
        first_shared = self.num_layers - num_kv_shared
        self._gemma4_kv_ref = {}
        prev_types = self.layer_types[:first_shared]
        for i in range(self.num_layers):
            if i >= first_shared and num_kv_shared > 0:
                lt = self.layer_types[i]
                last_idx = len(prev_types) - 1 - prev_types[::-1].index(lt)
                self._gemma4_kv_ref[i] = last_idx

        # Per-Layer Embeddings (PLE)
        ple_dim = getattr(self.llm_config, 'hidden_size_per_layer_input', None)
        self._gemma4_has_ple = ple_dim is not None and ple_dim > 0

    @override
    def init_config(self):
        super().init_config()
        self.tie_word_embeddings = True
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2

    @override
    def gen_embedding_lmhead_mlir(self):
        """Override to add per-layer inputs computation for PLE."""
        if not self._gemma4_has_ple:
            super().gen_embedding_lmhead_mlir()
            return

        tqdm.write("generate embedding, lm_head and per_layer_inputs mlir ...")
        embedding = self.model_info.weights[LlmList.EMBEDING]
        embedding_data = self.model.read(embedding + ".weight")

        if self.embedding_disk:
            self.gen_embedding_bin(embedding_data)
        else:
            embedding_weights = {embedding + ".weight": embedding_data}
            embedding_npz = "embedding_top_weights.npz"
            np.savez(embedding_npz, **embedding_weights)

        # lm_head
        lmhead = self.model_info.weights[LlmList.LMHEAD]
        lmhead_path = lmhead + ".weight"
        if self.tie_word_embeddings:
            lmhead_data = embedding_data
        else:
            lmhead_data = self.model.read(lmhead_path)
        lmhead_weights = {lmhead_path: lmhead_data}
        lmhead_npz = "lm_head_top_weights.npz"
        np.savez(lmhead_npz, **lmhead_weights)

        # PLE weights for embedding-level projection
        ple_dim = self.llm_config.hidden_size_per_layer_input
        num_layers = self.num_layers
        ple_embed_path = "model.language_model.embed_tokens_per_layer"
        ple_proj_path = "model.language_model.per_layer_model_projection"
        ple_proj_norm_path = "model.language_model.per_layer_projection_norm"

        ple_weights = {}
        ple_weights[ple_embed_path + ".weight"] = self.model.read(ple_embed_path + ".weight")
        ple_weights[ple_proj_path + ".weight"] = self.model.read(ple_proj_path + ".weight")
        ple_weights[ple_proj_norm_path + ".weight"] = self.model.read(
            ple_proj_norm_path + ".weight")
        ple_npz = "ple_top_weights.npz"
        np.savez(ple_npz, **ple_weights)

        per_layer_proj_scale = self.hidden_size**-0.5
        per_layer_input_scale = 2.0**-0.5
        ple_embed_scale = ple_dim**0.5

        def gen_embedding_by_length(name: str, seq_length: int):
            out_shape = [1, seq_length, self.hidden_size]
            ple_out_shape = [1, seq_length, num_layers, ple_dim]
            embedding_mlir = MLIRImporter([[1, seq_length]], [out_shape, ple_out_shape],
                                          name,
                                          self.platform,
                                          input_types=["INT32"],
                                          weight_file=f"../{embedding_npz}")
            input_op = embedding_mlir.create_input_op(
                self.get_loc("input_ids", embedding_mlir), 0)
            T = embedding_mlir.get_tensor_type
            L = lambda n: self.get_loc(n, embedding_mlir)
            ip = embedding_mlir.insert_point

            # Main embedding
            weight_op = embedding_mlir.create_weight_op(embedding + ".weight",
                                                        [self.vocab_size, self.hidden_size])
            new_op = top.GatherOp(T(out_shape),
                                  weight_op,
                                  input_op,
                                  axis=0,
                                  loc=L(name),
                                  ip=ip).output
            if self.scale_emb != 1.0:
                new_op = top.MulConstOp(T(out_shape),
                                        new_op,
                                        const_val=self.scale_emb,
                                        loc=L(name + ".scale"),
                                        ip=ip).output
            embedding_output = new_op

            # PLE: per-layer inputs from input embeddings
            ple_weight = embedding_mlir.create_weight_op(
                ple_proj_path + ".weight", [num_layers * ple_dim, self.hidden_size])
            ple_proj_op = top.MatMulOp(T([1, seq_length, num_layers * ple_dim]),
                                       embedding_output,
                                       ple_weight,
                                       embedding_mlir.none_op,
                                       loc=L("ple_proj"),
                                       ip=ip).output
            ple_proj_op = top.MulConstOp(T([1, seq_length, num_layers * ple_dim]),
                                         ple_proj_op,
                                         const_val=per_layer_proj_scale,
                                         loc=L("ple_proj.scale"),
                                         ip=ip).output
            ple_proj_op = top.ReshapeOp(T(ple_out_shape),
                                        ple_proj_op,
                                        shape=[1, seq_length, num_layers, ple_dim],
                                        loc=L("ple_proj.reshape"),
                                        ip=ip).output
            ple_norm_weight = embedding_mlir.create_weight_op(
                ple_proj_norm_path + ".weight", [1, 1, 1, ple_dim])
            ple_proj_op = top.RMSNormOp(T(ple_out_shape),
                                        ple_proj_op,
                                        ple_norm_weight,
                                        eps=self.rms_norm_eps,
                                        loc=L("ple_proj.norm"),
                                        ip=ip).output

            # PLE token identity lookup
            ple_embed_weight = embedding_mlir.create_weight_op(
                ple_embed_path + ".weight", [self.vocab_size, num_layers * ple_dim])
            ple_token_op = top.GatherOp(T([1, seq_length, num_layers * ple_dim]),
                                        ple_embed_weight,
                                        input_op,
                                        axis=0,
                                        loc=L("ple_token_lookup"),
                                        ip=ip).output
            ple_token_op = top.ReshapeOp(T(ple_out_shape),
                                         ple_token_op,
                                         shape=[1, seq_length, num_layers, ple_dim],
                                         loc=L("ple_token.reshape"),
                                         ip=ip).output

            # Combine: (token_identity + context_projection) * 1/sqrt(2)
            ple_combined = top.AddOp(T(ple_out_shape), [ple_token_op, ple_proj_op],
                                     loc=L("ple_combined"),
                                     ip=ip).output
            ple_combined = top.MulConstOp(T(ple_out_shape),
                                          ple_combined,
                                          const_val=per_layer_input_scale,
                                          loc=L("ple_combined.scale"),
                                          ip=ip).output

            embedding_mlir.create_return_op([embedding_output, ple_combined])
            mlir_txt = embedding_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # lm_head
        def gen_lm_head():
            name = "lm_head"
            out_shape = [[1, self.vocab_size]]
            if self.lmhead_with_topk:
                out_shape = [[1, 1]]
            if self.embedding_disk:
                out_shape.append([1, self.hidden_size])
            lmhead_mlir = MLIRImporter([[1, self.hidden_size]], out_shape,
                                       name,
                                       self.platform,
                                       input_types=["F32"],
                                       weight_file=f"../{lmhead_npz}")
            input_op = lmhead_mlir.create_input_op(
                self.get_loc("input_states", lmhead_mlir), 0)
            if self.lmhead_with_topk:
                weight_op = lmhead_mlir.create_weight_op(lmhead_path,
                                                         [self.hidden_size, self.vocab_size])
                new_op = top.MatMulOp(
                    lmhead_mlir.get_tensor_type([1, 1, self.vocab_size]),
                    input_op,
                    weight_op,
                    lmhead_mlir.none_op,
                    loc=self.get_loc("lm_head", lmhead_mlir),
                    ip=lmhead_mlir.insert_point).output
            else:
                weight_op = lmhead_mlir.create_weight_op(lmhead_path,
                                                         [self.vocab_size, self.hidden_size])
                new_op = top.MatMulOp(
                    lmhead_mlir.get_tensor_type([1, self.vocab_size]),
                    input_op,
                    weight_op,
                    lmhead_mlir.none_op,
                    loc=self.get_loc("lm_head", lmhead_mlir),
                    ip=lmhead_mlir.insert_point).output
            return_ops = [new_op]
            if self.embedding_disk:
                weight_op = lmhead_mlir.create_weight_op(embedding + ".weight",
                                                         [self.vocab_size, self.hidden_size])
                emb_op = top.GatherOp(
                    lmhead_mlir.get_tensor_type([1, self.hidden_size]),
                    weight_op,
                    lmhead_mlir.none_op,
                    axis=0,
                    loc=self.get_loc("embedding", lmhead_mlir),
                    ip=lmhead_mlir.insert_point).output
                return_ops.append(emb_op)
            lmhead_mlir.create_return_op(return_ops)
            mlir_txt = lmhead_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        gen_embedding_by_length("embedding", self.max_input_length)
        gen_embedding_by_length("embedding_cache", 1)
        gen_lm_head()

    @override
    def gen_vit_mlir(self):
        tqdm.write("generate vit mlir ...")
        name = "vit"
        vconfig = self.config.vision_config
        hidden_size = vconfig.hidden_size
        num_layers = vconfig.num_hidden_layers
        num_heads = vconfig.num_attention_heads
        kv_heads = vconfig.num_key_value_heads
        head_dim = vconfig.head_dim
        intermediate_size = vconfig.intermediate_size
        num_patches = vconfig.position_embedding_size
        patch_size = vconfig.patch_size
        mm_tokens = self.config.vision_soft_tokens_per_image
        hidden_act = vconfig.hidden_activation
        rms_norm_eps = vconfig.rms_norm_eps
        embed_dim = hidden_size

        vit_npz = "vit_top_weights.npz"
        top_path = "vision_tower"
        mm_projector = "model.embed_vision.embedding_projection"

        def save_weights():
            weights_dict = {}
            patch_embedder = f"model.{top_path}.patch_embedder"
            weights_dict[patch_embedder + ".input_proj.weight"] = self.model.read(
                patch_embedder + ".input_proj.weight")
            pos_embed = self.model.read(patch_embedder + ".position_embedding_table")
            weights_dict[patch_embedder + ".position_embedding_table"] = pos_embed[:num_patches, :]

            for idx in range(num_layers):
                layer_path = f"model.{top_path}.encoder.layers.{idx}"
                self.set_common_weight(f"{layer_path}.input_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.post_attention_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.pre_feedforward_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.post_feedforward_layernorm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.self_attn.q_norm", weights_dict,
                                       self.rmsnorm_type)
                self.set_common_weight(f"{layer_path}.self_attn.k_norm", weights_dict,
                                       self.rmsnorm_type)
                # Vision encoder uses Gemma4ClippableLinear with .linear.weight suffix
                self.set_linear_weight(f"{layer_path}.self_attn.q_proj.linear", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.k_proj.linear", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.v_proj.linear", weights_dict)
                self.set_linear_weight(f"{layer_path}.self_attn.o_proj.linear", weights_dict)
                self.set_linear_weight(f"{layer_path}.mlp.gate_proj.linear", weights_dict)
                self.set_linear_weight(f"{layer_path}.mlp.up_proj.linear", weights_dict)
                self.set_linear_weight(f"{layer_path}.mlp.down_proj.linear", weights_dict)
            weights_dict[mm_projector + ".weight"] = self.model.read(mm_projector + ".weight")
            np.savez(vit_npz, **weights_dict)

        save_weights()

        # Image input: variable resolution, use a square image with patches_per_side
        patches_per_side = int(num_patches**0.5)
        image_size = patches_per_side * patch_size
        in_shape = [1, 3, image_size, image_size]
        out_shape = [1, num_patches, self.hidden_size]
        hidden_shape = [1, num_patches, embed_dim]
        vit_mlir = MLIRImporter([in_shape], [out_shape],
                                name,
                                self.platform, ["F32"],
                                weight_file=f"../{vit_npz}")
        ip = vit_mlir.insert_point
        T = vit_mlir.get_tensor_type
        L = lambda name: self.get_loc(name, vit_mlir)

        in_op = vit_mlir.create_input_op(L('pixel_values'), 0)
        # Patch embedding: permute and flatten
        new_op = top.ReshapeOp(
            T([1, 3, patches_per_side, patch_size, patches_per_side, patch_size]),
            in_op, loc=L("pixel_reshape"), ip=ip).output
        new_op = top.PermuteOp(
            T([1, patches_per_side, patches_per_side, 3, patch_size, patch_size]),
            new_op, order=[0, 2, 4, 1, 3, 5],
            loc=L("pixel_transpose"), ip=ip).output
        new_op = top.ReshapeOp(T([1, num_patches, 3 * patch_size * patch_size]),
                               new_op, loc=L("pixel_reshape2"), ip=ip).output

        patch_weight = vit_mlir.create_weight_op(
            f"model.{top_path}.patch_embedder.input_proj.weight",
            [3 * patch_size * patch_size, embed_dim])
        new_op = top.MatMulOp(T(hidden_shape),
                              new_op, patch_weight, vit_mlir.none_op,
                              loc=L("patch_embedder"), ip=ip).output

        pos_weight = vit_mlir.create_weight_op(
            f"model.{top_path}.patch_embedder.position_embedding_table",
            [1, num_patches, embed_dim])
        new_op = top.AddOp(T(hidden_shape), [new_op, pos_weight],
                           loc=L("position_embedding.add"), ip=ip).output

        for idx in range(num_layers):
            layer_path = f"model.{top_path}.encoder.layers.{idx}"
            residual_op = new_op
            new_op = self.rms_norm(vit_mlir, new_op, f"{layer_path}.input_layernorm",
                                   eps=rms_norm_eps)
            q_op = self.linear(vit_mlir, f"{layer_path}.self_attn.q_proj.linear", new_op,
                               [embed_dim, embed_dim], hidden_shape)
            k_op = self.linear(vit_mlir, f"{layer_path}.self_attn.k_proj.linear", new_op,
                               [embed_dim, embed_dim], hidden_shape)
            v_op = self.linear(vit_mlir, f"{layer_path}.self_attn.v_proj.linear", new_op,
                               [embed_dim, embed_dim], hidden_shape)

            new_shape = [1, num_patches, num_heads, head_dim]
            q_op = top.ReshapeOp(T(new_shape), q_op,
                                  loc=L(f"{layer_path}.self_attn.q_reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(new_shape), k_op,
                                  loc=L(f"{layer_path}.self_attn.k_reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(new_shape), v_op,
                                  loc=L(f"{layer_path}.self_attn.v_reshape"), ip=ip).output

            q_op = self.rms_norm(vit_mlir, q_op, f"{layer_path}.self_attn.q_norm",
                                 eps=rms_norm_eps)
            k_op = self.rms_norm(vit_mlir, k_op, f"{layer_path}.self_attn.k_norm",
                                 eps=rms_norm_eps)

            fa_op = top.FAttentionOp(T(hidden_shape),
                                     q_op, k_op, v_op,
                                     vit_mlir.none_op, vit_mlir.none_op,
                                     scale=head_dim**-0.5,
                                     batch=1,
                                     q_head=num_heads,
                                     kv_head=kv_heads,
                                     dim=head_dim,
                                     mq=num_patches,
                                     mk=num_patches,
                                     keep_dims=False,
                                     loc=L(f"{layer_path}.fattention"),
                                     ip=ip).output

            o_op = self.linear(vit_mlir, f"{layer_path}.self_attn.o_proj.linear", fa_op,
                               [embed_dim, embed_dim], hidden_shape)
            o_op = self.rms_norm(vit_mlir, o_op, f"{layer_path}.post_attention_layernorm",
                                 eps=rms_norm_eps)
            new_op = top.AddOp(T(hidden_shape), [residual_op, o_op],
                               loc=L(f"{layer_path}.residual_add"), ip=ip).output

            residual_op = new_op
            new_op = self.rms_norm(vit_mlir, new_op, f"{layer_path}.pre_feedforward_layernorm",
                                   eps=rms_norm_eps)
            gate_op = self.linear(vit_mlir, f"{layer_path}.mlp.gate_proj.linear", new_op,
                                  [embed_dim, intermediate_size],
                                  [1, num_patches, intermediate_size])
            act_op = self.activate(vit_mlir, gate_op, hidden_act, layer_path)
            up_op = self.linear(vit_mlir, f"{layer_path}.mlp.up_proj.linear", new_op,
                                [embed_dim, intermediate_size],
                                [1, num_patches, intermediate_size])
            mlp_op = top.MulOp(T([1, num_patches, intermediate_size]), [act_op, up_op],
                               loc=L(f"{layer_path}.mlp.mul"), ip=ip).output
            down_op = self.linear(vit_mlir, f"{layer_path}.mlp.down_proj.linear", mlp_op,
                                  [intermediate_size, embed_dim], hidden_shape)
            down_op = self.rms_norm(vit_mlir, down_op, f"{layer_path}.post_feedforward_layernorm",
                                    eps=rms_norm_eps)
            new_op = top.AddOp(T(hidden_shape), [residual_op, down_op],
                               loc=L(f"{layer_path}.mlp.add"), ip=ip).output

        # mm projector
        mm_weight = vit_mlir.create_weight_op(mm_projector + ".weight",
                                              [embed_dim, self.hidden_size])
        new_op = top.MatMulOp(T([1, num_patches, self.hidden_size]),
                              new_op, mm_weight, vit_mlir.none_op,
                              loc=L("mm_projector.matmul"), ip=ip).output
        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)
