

Lattice_Transformer_SeqLabel(
  (bert_embedding): BertEmbedding(
    (dropout_layer): Dropout(p=0, inplace=False)
    (model): _BertWordModel(
      (encoder): BertModel(
        (embeddings): BertEmbeddings(
          (word_embeddings): Embedding(21128, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
          (layer): ModuleList(
            (0): BertLayer(
              (attention): BertAttention(
                (self): BertSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): BertSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): BertIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
              )
              (output): BertOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1):
            ...
            (11):
          )
        )
        (pooler): BertPooler(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (activation): Tanh()
        )
      )
    )
  )
  (lattice_embed): StaticEmbedding(
    (dropout_layer): Dropout(p=0, inplace=False)
    (embedding): Embedding(17814, 50, padding_idx=0)
    (dropout): MyDropout()
  )
  (bigram_embed): StaticEmbedding(
    (dropout_layer): Dropout(p=0, inplace=False)
    (embedding): Embedding(41262, 50, padding_idx=0)
    (dropout): MyDropout()
  )
  (embed_dropout): MyDropout()
  (gaz_dropout): MyDropout()
  (char_proj): Linear(in_features=868, out_features=160, bias=True)
  (lex_proj): Linear(in_features=50, out_features=160, bias=True)
  (encoder): Transformer_Encoder(
    (four_pos_fusion_embedding): Four_Pos_Fusion_Embedding(
      (pos_fusion_forward): Sequential(
        (0): Linear(in_features=320, out_features=160, bias=True)
        (1): ReLU(inplace=True)
      )
    )
    (layer_0): Transformer_Encoder_Layer(
      (four_pos_fusion_embedding): Four_Pos_Fusion_Embedding(
        (pos_fusion_forward): Sequential(
          (0): Linear(in_features=320, out_features=160, bias=True)
          (1): ReLU(inplace=True)
        )
      )
      (layer_preprocess): Layer_Process()
      (layer_postprocess): Layer_Process(
        (layer_norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
      )
      (attn): MultiHead_Attention_Lattice_rel_save_gpumm(
        (w_k): Linear(in_features=160, out_features=160, bias=True)
        (w_q): Linear(in_features=160, out_features=160, bias=True)
        (w_v): Linear(in_features=160, out_features=160, bias=True)
        (w_r): Linear(in_features=160, out_features=160, bias=True)
        (w_final): Linear(in_features=160, out_features=160, bias=True)
        (dropout): MyDropout()
      )
      (ff): Positionwise_FeedForward(
        (w0): Linear(in_features=160, out_features=480, bias=True)
        (w1): Linear(in_features=480, out_features=160, bias=True)
        (dropout): MyDropout()
        (dropout_2): MyDropout()
        (activate): ReLU(inplace=True)
      )
    )
    (layer_preprocess): Layer_Process()
  )
  (output_dropout): MyDropout()
  (output): Linear(in_features=160, out_features=19, bias=True)
  (crf): ConditionalRandomField()
  (loss_func): CrossEntropyLoss()
)

