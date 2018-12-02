import numpy as np
import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class CNNRegressor(Chain):

    def __init__(self, n_words, embed_size, n_filters, filters=[2,3,4]):
        super(CNNRegressor, self).__init__()
        with self.init_scope():
            self.embed_size = embed_size
            self.embed = L.EmbedID(n_words, self.embed_size)
            self.conv = []
            self.bn = []
            for i, k in enumerate(filters):
                setattr(self, "conv{}".format(i+1), L.Convolution2D(None, n_filters, (k, embed_size)))
                self.conv.append(getattr(self, "conv{}".format(i+1)))
                setattr(self, "bn{}".format(i+1), L.BatchNormalization(n_filters))
                self.bn.append(getattr(self, "bn{}".format(i+1)))
            self.bn_fc = L.BatchNormalization(n_filters * len(filters))
            self.fc = L.Linear(None, 1)

    def __call__(self, x):
        n_docs = len(x)
        #docs -> N x 2D_Arr(length x embed_size)
        emb = self.embed(x)
        emb = emb.reshape(n_docs,1,-1,self.embed_size)    #NCHW
        #Convolution and Activation(relu)
        self.h_conv = [F.tanh(self.bn[i](conv(emb))) for i, conv in enumerate(self.conv)]
        #Max_pooling_2D
        self.h_pool = [F.dropout(F.max_pooling_2d(h_conv, (h_conv.shape[2],1)), ratio=.5) for h_conv in self.h_conv]
        #Concat
        self.concat = F.concat(self.h_pool, axis=2).reshape(n_docs,-1)
        #Linear
        y = F.relu(self.fc(self.bn_fc(self.concat)))
        return y


class GatedLinearUnit(Chain):

    def __init__(self, input_size, output_size, kernel_size=3, pad=None):
        super(GatedLinearUnit, self).__init__()
        with self.init_scope():
            self.input_size = input_size    #Embed_size
            self.output_size = output_size	#Encode_size
            self.kernel_size = kernel_size
            self.pad = kernel_size // 2
            if not pad is None:
                self.pad = pad

            self.conv_A = L.Convolution2D(None, self.output_size, (self.kernel_size, self.input_size+2*self.pad), pad=self.pad)
            self.conv_B = L.Convolution2D(None, self.output_size, (self.kernel_size, self.input_size+2*self.pad), pad=self.pad)

    def __call__(self, x):
        # EmbedID (N, length, input_size) -> NCHW (N, 1, length, input_size)
        N, l, *_ = x.shape
        if not len(_) == 1:
            print("x.shape is {}".format(x.shape))
            print("Please input in the form of (N, length, input_size)")
        x = x.reshape(N,1,l,self.input_size)

        # Convolution
        A = self.conv_A(x)              # (N, output_size, length, 1)
        B = F.sigmoid(self.conv_B(x))   # (N, output_size, length, 1)

        # Reshape
        self.A = A.reshape(N, self.output_size, -1) # (N, output_size, length)
        self.B = B.reshape(N, self.output_size, -1) # (N, output_size, length)

        # Output
        self.y = self.A * self.B
        self.y = F.transpose(self.y, axes=(0,2,1))  # (N, length, output_size)
        return self.y


class GatedCNNRegressor(Chain):

    def __init__(self, n_words, embed_size=50, hidden_size=[256,256,256]):
        super(GatedCNNRegressor, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_words, embed_size)
            self.input_size = [embed_size] + hidden_size
            for i, h in enumerate(hidden_size):
                setattr(self, "glu_{}".format(i+1), GatedLinearUnit(input_size=self.input_size[i], output_size=h))
            self.fc1 = L.Linear(hidden_size[-1], hidden_size[-1])
            self.bn = L.BatchNormalization(hidden_size[-1])
            self.fc2 = L.Linear(hidden_size[-1], 1)

    def __call__(self, x):
        x = self.embed(x)
        for i in range(1, len(self.input_size)):
            x = getattr(self, "glu_{}".format(i))(x)
        self.x_softmax = F.softmax(x, axis=2)
        self.x_sum = F.sum(self.x_softmax * x, axis=1)
        self.fc_1 = F.relu(self.bn(self.fc1(self.x_sum)))
        self.y = self.fc2(F.dropout(self.fc_1, ratio=.5))
        return self.y
    

class Attention(Chain):

    def __init__(self, attn_size=128):
        super(Attention, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, attn_size, nobias=True)
            self.l2 = L.Linear(attn_size, 1, nobias=True)

    def __call__(self, xs):
        concat_xs = F.concat(xs, axis=0)                # (n, s, h) -> (n*s, h)
        attn = self.l2(F.tanh(self.l1(concat_xs)))      # (n*s, h) -> (n*s, 1)
        split_attn = F.split_axis(  attn,               # (n*s,1) -> (n, s, 1)
                                    np.cumsum([len(x) for x in xs])[:-1],
                                    axis=0)
        split_attn_pad = F.pad_sequence(split_attn,     # (n, s, 1) -> (n, l, 1) 
                                        padding=-1024.) # exp(-1024.) = 0.
        return F.softmax(split_attn_pad, axis=1)


class BiLSTMwithAttentionRegressor(Chain):

    def __init__(self, n_words, n_layers=1, embed_size=128, hidden_size=128, dropout=.3, initialW=None):
        super(BiLSTMwithAttentionRegressor, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_words, embed_size, initialW=initialW)
            if not initialW is None:
                self.embed.disable_update()
            self.bilstm = L.NStepBiLSTM(n_layers=n_layers,
                                        in_size=embed_size,
                                        out_size=hidden_size,
                                        dropout=dropout)
            self.attn = Attention(attn_size=hidden_size*2)
            self.fc1 = L.Linear(hidden_size*2, hidden_size*2)
            self.fc2 = L.Linear(hidden_size*2, 1)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

    def __call__(self, xs):
        emb = self.seq_embed(self.embed, xs)                # (n, s) -> (n, s, e)
        last_h, last_c, ys = self.bilstm(None, None, emb)   # (n, s, e) -> ys(n, s, h*2)
        self.attention = self.attn(ys)                      # (n, s, h*2) -> (n, l, 1)
        pad_ys = F.pad_sequence(ys, padding=0.)             # (n, s, h*2) -> (n, l, h*2)
        h = F.matmul(pad_ys.transpose(0,2,1),               # (n, l, h*2) -> (n, h*2, l)
                     self.attention                         # (n, h*2, l)(n, l, 1) -> (n, h*2, 1)
                    ).reshape(-1, self.hidden_size*2)       # (n, h*2, 1) -> (n, h*2)
        fc1 = F.relu(self.fc1(h))                           # (n, h*2) -> (n, h*2)
        fc2 = self.fc2(F.dropout(fc1, ratio=.5))            # (n, h*2) -> (n, 1)
        return fc2

    def seq_embed(self, embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return exs


class BiGRUwithAttentionRegressor(Chain):

    def __init__(self, n_words, n_layers=1, embed_size=128, hidden_size=128, dropout=.3, initialW=None):
        super(BiGRUwithAttentionRegressor, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_words, embed_size, initialW=initialW)
            if not initialW is None:
                self.embed.disable_update()
            self.bigru = L.NStepBiGRU(  n_layers=n_layers,
                                        in_size=embed_size,
                                        out_size=hidden_size,
                                        dropout=dropout)
            self.attn = Attention(attn_size=hidden_size*2)
            self.fc1 = L.Linear(hidden_size*2, hidden_size*2)
            self.brn = L.BatchNormalization(hidden_size*2)
            self.fc2 = L.Linear(hidden_size*2, 1)
        self.n_layers = n_layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout

    def __call__(self, xs):
        emb = self.seq_embed(self.embed, xs)                # (n, s) -> (n, s, e)
        last_h, ys = self.bigru(None, emb)                  # (n, s, e) -> ys(n, s, h*2)
        self.attention = self.attn(ys)                      # (n, s, h*2) -> (n, l, 1)
        pad_ys = F.pad_sequence(ys, padding=0.)             # (n, s, h*2) -> (n, l, h*2)
        h = F.matmul(pad_ys.transpose(0,2,1),               # (n, l, h*2) -> (n, h*2, l)
                     self.attention                         # (n, h*2, l)(n, l, 1) -> (n, h*2, 1)
                    ).reshape(-1, self.hidden_size*2)       # (n, h*2, 1) -> (n, h*2)
        fc1 = F.relu(self.brn(self.fc1(h)))                 # (n, h*2) -> (n, h*2)
        fc2 = self.fc2(F.dropout(fc1, ratio=.5))            # (n, h*2) -> (n, 1)
        return fc2

    def seq_embed(self, embed, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = embed(F.concat(xs, axis=0))
        exs = F.split_axis(ex, x_section, 0)
        return exs


class MyRegressor(Chain):

    def __init__(self, predictor):
        super(MyRegressor, self).__init__(predictor=predictor)

    def __call__(self, x, y):
        pred = self.predictor(x)
        loss = F.mean_squared_error(pred, y)
        r2 = F.r2_score(pred, y)
        report({'loss':loss, 'r2':r2}, self)
        return loss
