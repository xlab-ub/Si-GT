"""Biased Multi-head Attention"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BiasedMHA(nn.Module):
    r"""Dense Multi-Head Attention Module with Graph Attention Bias.

    Compute attention between nodes with attention bias obtained from graph
    structures, as introduced in `Do Transformers Really Perform Bad for
    Graph Representation? <https://arxiv.org/pdf/2106.05234>`__

    .. math::

        \text{Attn}=\text{softmax}(\dfrac{QK^T}{\sqrt{d}} \circ b)

    :math:`Q` and :math:`K` are feature representations of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\circ`.

    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: True.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    attn_drop : float, optional
        Dropout probability on attention weights. Defalt: 0.1.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import BiasedMHA

    >>> ndata = th.rand(16, 100, 512)
    >>> bias = th.rand(16, 100, 100, 8)
    >>> net = BiasedMHA(feat_size=512, num_heads=8)
    >>> out = net(ndata, bias)
    """

    def __init__(
        self,
        feat_size,
        num_heads,
        bias=True,
        attn_bias_type="add",
        attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of projection matrices, the same settings as in
        the original implementation of the paper.
        """
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None):
        """Forward computation with IIN-Attn.

        Parameters
        ----------
        ndata : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, feat_size),
            where N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias matrix Φ for IIN-Attn.
            Shape: (batch_size, N, N, num_heads).
        attn_mask : torch.Tensor, optional
            The attention mask to avoid computation on invalid positions.
            Shape: (batch_size, N, N).
            For rows corresponding to non-existent nodes, at least one entry 
            should be False to avoid NaNs in softmax.

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, feat_size).
        """
        # 1. Project inputs to Q, K, V
        q_h = self.q_proj(ndata).transpose(0, 1)  # (N, batch_size, dim)
        k_h = self.k_proj(ndata).transpose(0, 1)  # (N, batch_size, dim)
        v_h = self.v_proj(ndata).transpose(0, 1)  # (N, batch_size, dim)
        
        bsz, N, _ = ndata.shape

        # 2. Reshape for multi-head attention
        # Q: (batch_size*num_heads, N, head_dim)
        q_h = q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1) * self.scaling
        # K: (batch_size*num_heads, head_dim, N)
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)
        # V: (batch_size*num_heads, N, head_dim)
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 3. Compute scaled dot-product attention logits
        attn_logits = th.bmm(q_h, k_h)  # (batch_size*num_heads, N, N)
        # Reshape to (N, N, bsz, num_heads), then permute to (bsz, num_heads, N, N)
        attn_logits = (
            attn_logits.transpose(0, 2)
                    .reshape(N, N, bsz, self.num_heads)
                    .permute(2, 3, 0, 1)
        )

        # 4. Incorporate IIN-Attn bias (Φ)
        # Expected attn_bias shape: (batch_size, N, N, num_heads)
        if attn_bias is not None:
            # Permute attn_bias to (batch_size, num_heads, N, N)
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            # Element-wise add the bias to the logits
            attn_logits += attn_bias

        # 5. Apply attention mask if provided
        if attn_mask is not None:
            # Expand to (batch_size, 1, N, N) for compatibility
            attn_mask = attn_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(attn_mask.bool(), float("-inf"))

        # 6. Softmax over the last dimension (N)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Compute weighted sum of value vectors
        # Flatten back to (batch_size*num_heads, N, N) for bmm
        attn_weights = attn_weights.reshape(bsz * self.num_heads, N, N)
        # attn: (batch_size*num_heads, N, head_dim)
        attn = th.bmm(attn_weights, v_h)

        # 8. Reshape and apply output projection
        # -> (N, batch_size, num_heads * head_dim) -> (batch_size, N, feat_size)
        attn = attn.transpose(0, 1).reshape(N, bsz, self.num_heads * self.head_dim).transpose(0, 1)
        attn = self.out_proj(attn)

        return attn


class GraphormerLayer(nn.Module):
    r"""Graphormer Layer with Dense Multi-Head Attention, as introduced
    in `Do Transformers Really Perform Bad for Graph Representation?
    <https://arxiv.org/pdf/2106.05234>`__

    Parameters
    ----------
    feat_size : int
        Feature size.
    hidden_size : int
        Hidden size of feedforward layers.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    norm_first : bool, optional
        If True, it performs layer normalization before attention and
        feedforward operations. Otherwise, it applies layer normalization
        afterwards. Default: False.
    dropout : float, optional
        Dropout probability. Default: 0.1.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.1.
    activation : callable activation layer, optional
        Activation function. Default: nn.ReLU().

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import GraphormerLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size = 512
    >>> num_heads = 8
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> bias = th.rand(batch_size, num_nodes, num_nodes, num_heads)
    >>> net = GraphormerLayer(
            feat_size=feat_size,
            hidden_size=2048,
            num_heads=num_heads
        )
    >>> out = net(nfeat, bias)
    """

    def __init__(
        self,
        feat_size,
        hidden_size,
        num_heads,
        attn_bias_type="add",
        norm_first=False,
        dropout=0.1,
        attn_dropout=0.1,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMHA(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):
        """Forward computation.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where invalid positions are indicated by `True` values.
            Shape: (batch_size, N, N). Note: For rows corresponding to
            unexisting nodes, make sure at least one entry is set to `False` to
            prevent obtaining NaNs with softmax.

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)]
        """
        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat = self.attn(nfeat, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat

class BiasedMHA_visual_attn(nn.Module):
    r"""Dense Multi-Head Attention Module with Graph Attention Bias.

    Compute attention between nodes with attention bias obtained from graph
    structures, as introduced in `Do Transformers Really Perform Bad for
    Graph Representation? <https://arxiv.org/pdf/2106.05234>`__

    .. math::

        \text{Attn}=\text{softmax}(\dfrac{QK^T}{\sqrt{d}} \circ b)

    :math:`Q` and :math:`K` are feature representations of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\circ`.

    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: True.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    attn_drop : float, optional
        Dropout probability on attention weights. Defalt: 0.1.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import BiasedMHA

    >>> ndata = th.rand(16, 100, 512)
    >>> bias = th.rand(16, 100, 100, 8)
    >>> net = BiasedMHA(feat_size=512, num_heads=8)
    >>> out = net(ndata, bias)
    """

    def __init__(
        self,
        feat_size,
        num_heads,
        bias=True,
        attn_bias_type="add",
        attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters of projection matrices, the same settings as in
        the original implementation of the paper.
        """
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None):
        """Forward computation with IIN-Attn.

        Parameters
        ----------
        ndata : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, feat_size),
            where N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias matrix Φ for IIN-Attn.
            Shape: (batch_size, N, N, num_heads).
        attn_mask : torch.Tensor, optional
            The attention mask to avoid computation on invalid positions.
            Shape: (batch_size, N, N).
            For rows corresponding to non-existent nodes, at least one entry 
            should be False to avoid NaNs in softmax.

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, feat_size).
        """
        # 1. Project inputs 
        q_h = self.q_proj(ndata).transpose(0, 1)  
        k_h = self.k_proj(ndata).transpose(0, 1) 
        v_h = self.v_proj(ndata).transpose(0, 1)  
        
        bsz, N, _ = ndata.shape

        # 2. Reshape for multi-head attention
        q_h = q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1) * self.scaling
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 3. Compute scaled dot-product attention logits
        attn_logits = th.bmm(q_h, k_h)  # (batch_size*num_heads, N, N)
        attn_logits = (
            attn_logits.transpose(0, 2)
                    .reshape(N, N, bsz, self.num_heads)
                    .permute(2, 3, 0, 1)
        )

        # 4. Incorporate IIN-Attn bias (Φ)
        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            attn_logits += attn_bias

        # 5. Apply attention mask if provided
        if attn_mask is not None:
            # Expand to (batch_size, 1, N, N) for compatibility
            attn_mask = attn_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(attn_mask.bool(), float("-inf"))

        # 6. Softmax over the last dimension (N)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Compute weighted sum of value vectors
        attn_weights = attn_weights.reshape(bsz * self.num_heads, N, N)
        attn = th.bmm(attn_weights, v_h)

        # 8. Reshape and apply output projection
        attn = attn.transpose(0, 1).reshape(N, bsz, self.num_heads * self.head_dim).transpose(0, 1)
        attn = self.out_proj(attn)

        return attn, attn_weights.reshape(bsz, self.num_heads, N, N)


class GraphormerLayer_visual_attn(nn.Module):
    r"""Graphormer Layer with Dense Multi-Head Attention, as introduced
    in `Do Transformers Really Perform Bad for Graph Representation?
    <https://arxiv.org/pdf/2106.05234>`__

    Parameters
    ----------
    feat_size : int
        Feature size.
    hidden_size : int
        Hidden size of feedforward layers.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    norm_first : bool, optional
        If True, it performs layer normalization before attention and
        feedforward operations. Otherwise, it applies layer normalization
        afterwards. Default: False.
    dropout : float, optional
        Dropout probability. Default: 0.1.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.1.
    activation : callable activation layer, optional
        Activation function. Default: nn.ReLU().

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import GraphormerLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size = 512
    >>> num_heads = 8
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> bias = th.rand(batch_size, num_nodes, num_nodes, num_heads)
    >>> net = GraphormerLayer(
            feat_size=feat_size,
            hidden_size=2048,
            num_heads=num_heads
        )
    >>> out = net(nfeat, bias)
    """

    def __init__(
        self,
        feat_size,
        hidden_size,
        num_heads,
        attn_bias_type="add",
        norm_first=False,
        dropout=0.1,
        attn_dropout=0.1,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMHA_visual_attn(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):
        """Forward computation.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where invalid positions are indicated by `True` values.
            Shape: (batch_size, N, N). Note: For rows corresponding to
            unexisting nodes, make sure at least one entry is set to `False` to
            prevent obtaining NaNs with softmax.

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)]
        """
        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat, attn_weights = self.attn(nfeat, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat, attn_weights


