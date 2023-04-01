import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def mask_tensor_random(self, tensor, percentage, seasonal, trend, trend_y, seasonal_y):

        shape = tensor.shape
        shape_y = seasonal_y.shape

        num_elements = shape[1]
        num_elements_y = shape_y[1]

        if percentage < 0 or percentage > 100:
                raise ValueError("Percentage value should be between 0 and 100")
        if percentage == 0:
            return tensor, seasonal, seasonal_y

        num_zeros = int((100 - percentage) / 100 * num_elements)
        num_zeros_y = int((100 - percentage) / 100 * num_elements_y)

        mask = torch.zeros((1, num_elements, 1), dtype=torch.float32)
        mask_y = torch.zeros((1, num_elements_y, 1), dtype=torch.float32)

        mask[0, torch.randperm(num_elements)[:num_zeros], 0] = 1
        mask_y[0, torch.randperm(num_elements_y)[:num_zeros_y], 0] = 1

        masked_tensor = tensor * mask.to(tensor.device)
        masked_season = seasonal * mask.to(seasonal.device)
        masked_trend = trend * mask.to(seasonal.device)
        masked_season_y = seasonal_y * mask_y.to(seasonal.device)
        masked_trend_y = trend_y * mask_y.to(seasonal.device)

        return masked_tensor, masked_season, masked_trend, masked_trend_y, masked_season_y

    def mask_tensor_random_all(self, tensor, percentage):

        shape = tensor.shape
        num_elements = shape[1]
        num_zeros = int((100 - percentage) / 100 * num_elements)
        mask = torch.zeros((1, num_elements, 1), dtype=torch.float32)
        mask[0, torch.randperm(num_elements)[:num_zeros], 0] = 1
        masked_tensor = tensor * mask.to(tensor.device)

        return masked_tensor

    def mask_tensor_fix(self, enc, dec, dec_s, percentage):

        if percentage < 0 or percentage > 100:
                raise ValueError("Percentage value should be between 0 and 100")
        if percentage == 0:
            return enc, dec, dec_s

        masked_size_enc = int(enc.shape[1] * percentage / 100)
        masked_size_dec = int(dec.shape[1] * percentage / 100)
        mask_enc = torch.zeros(enc.shape)
        mask_dec = torch.zeros(dec.shape)

        mask_enc[:, masked_size_enc:, :] = 1
        mask_dec[:, masked_size_dec:, :] = 1

        mask_enc = mask_enc.to(enc.device)
        mask_dec = mask_dec.to(dec.device)

        return enc*mask_enc, dec_s*mask_enc, dec*mask_dec

    @staticmethod
    def get_rand_mask(shape, mu=5, std=3):
        batch_size = shape[0]
        len_ds = shape[1]
        feature_dim = shape[2]
        # print("batch_size:",batch_size,"len_ds:", len_ds,"feature_dim:",feature_dim)
        span_size = int(np.random.normal(mu, std)) + 5
        # print("spansize:",span_size)
        if span_size < 1:
            span_size = 1
        n_spans = int(len_ds / span_size)
        mask = np.random.choice([False, True], [batch_size, n_spans, feature_dim], p=[0.8, 0.2])

        if not np.all(np.any(mask, axis=1)):
            rand_true = np.random.randint(n_spans - 1)
            mask[:, rand_true, :] = True

        mask = np.repeat(mask, span_size, axis=1)
        ones = np.zeros([batch_size, len_ds - n_spans * span_size, feature_dim], dtype=bool)
        mask = np.concatenate([mask, ones], axis=1)
        return torch.tensor(mask, dtype=bool)

    def mask_init(self, seasonal_init):

        seasonal_init_tmp = seasonal_init.detach().cpu().numpy()
        seasonal_init_tmp = torch.tensor(seasonal_init_tmp)
        shape = seasonal_init.shape
        mask = self.get_rand_mask(shape)
        zeros = torch.zeros(seasonal_init.shape, dtype=seasonal_init.dtype)

        seasonal_init_masked = torch.where(mask, seasonal_init_tmp, zeros)

        seasonal_init_masked = seasonal_init_masked.float().to(device)

        mask = mask.to(device)
        dec_input = seasonal_init * mask

        return torch.tensor(dec_input), mask




    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pre_train=False, batch_y=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if pre_train:

            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, 6, 1)
            zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()

            #masking whole time series
            #x_enc = self.mask_tensor_random_all(x_enc, 20)

            seasonal_init, trend_init = self.decomp(x_enc)
            seasonal_init_y, trend_init_y = self.decomp(batch_y)


            #encoder masking
            x_enc, _, _, _, seasonal_init_y = self.mask_tensor_random(x_enc, 30, seasonal_init, trend_init, trend_init_y, seasonal_init_y)
            #x_enc, _, seasonal_init_y = self.mask_tensor_fix(x_enc, seasonal_init_y, seasonal_init, 40)

            #decoder masking
            #_, seasonal_init, _, _, seasonal_init_y = self.mask_tensor_random(x_enc, 40, seasonal_init, trend_init, trend_init_y, seasonal_init_y)

            #_, seasonal_init, seasonal_init_y = self.mask_tensor_fix(x_enc, seasonal_init_y, seasonal_init, 20)

            trend_init = torch.cat([trend_init[:, :, :], trend_init_y[:, -(self.pred_len - self.label_len):, :]], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, :, :], seasonal_init_y[:, -(self.pred_len - self.label_len):, :]], dim=1)

            enc_out = self.enc_embedding(x_enc, x_mark_enc)

            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                     trend=trend_init)
            dec_out = trend_part + seasonal_part

            if self.output_attention:
                return dec_out[:, :, :], attns
            else:
                return dec_out[:, :, :]  # [B, L, D]

        else:
            # decomp init
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
            seasonal_init, trend_init = self.decomp(x_enc)
            seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

            # enc
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
            # dec
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                     trend=trend_init)
            # final
            dec_out = trend_part + seasonal_part

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], attns
            else:
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        version = 'Fourier'
        # version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)
