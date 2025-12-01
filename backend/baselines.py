import torch.nn as nn
import torch.nn.functional as F
import torch
# import flashinfer
import math
import tensorrt as trt
import tempfile
import os
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

class Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):

        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # q = q.transpose(0, 1)

        # scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)
        
        # output = torch.matmul(weights, self.cache_V)
        # output = output.view(self.M, self.H * self.D)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        # weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        # output = output.reshape(self.M, self.H * self.D)
        return output

class Vanilla_GQA(nn.Module):
    def __init__(self, M, N, D, P, N_kv, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.N_kv = N_kv
        self.H_kv = N_kv // D
        self.group_size = self.H // self.H_kv
        self.kv_idx = torch.arange(self.H, device=device) // self.group_size
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):

        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q = qkv[:, :self.N]
        k = qkv[:, self.N:self.N+self.N_kv]
        v = qkv[:, self.N+self.N_kv:]
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H_kv, self.D)
        v = v.view(self.M, self.H_kv, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_expanded = k.repeat_interleave(self.group_size, dim=0)
        v_expanded = v.repeat_interleave(self.group_size, dim=0)

        self.cache_K[:, self.P:self.P+self.M, :] = k_expanded
        self.cache_V[:, self.P:self.P+self.M, :] = v_expanded

        k_cache = self.cache_K
        v_cache = self.cache_V

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        # output = output.reshape(self.M, self.H * self.D)
        return output

class PreNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        # q = torch.matmul(X_norm, self.W_q)
        # k = torch.matmul(X_norm, self.W_k)
        # v = torch.matmul(X_norm, self.W_v)

        qkv = torch.matmul(X_norm, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        # output = output.reshape(self.M, self.H * self.D)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output

class KeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.noise = noise.to(device=device, dtype=dtype)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)
        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V


        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        # weights = F.softmax(scores, dim=-1)

        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        perturb_exp = torch.exp(perturb)
        perturb_sum = torch.sum(perturb_exp, dim=-1, keepdim=True)
        perturb_out = perturb_exp / perturb_sum

        # perturb_out = F.softmax(perturb, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output, perturb_out

class NormKeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.noise = noise.to(device=device, dtype=dtype)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X_norm, self.W_q)
        k = torch.matmul(X_norm, self.W_k)
        v = torch.matmul(X_norm, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        weights = F.softmax(scores, dim=-1)
        perturb_out = F.softmax(perturb, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output

class QKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)
        self.cache_K[:, self.P:self.P+self.M, :] = k_norm
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)
        scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output

class NormQKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X_norm, self.W_q)
        k = torch.matmul(X_norm, self.W_k)
        v = torch.matmul(X_norm, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output


class RoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # scores = torch.matmul(q, k_cache.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output, weights_sum, weights_sqr_sum


class NormRoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X_norm, self.W_q)
        k = torch.matmul(X_norm, self.W_k)
        v = torch.matmul(X_norm, self.W_v)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)
        
        output = torch.matmul(weights, v_cache)
        output = output.view(self.M, self.H * self.D)

        return output

class SimpleAttention(nn.Module):
    """Simple manual attention implementation for comparison"""
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        seq_len  = X.shape[0]
        
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(seq_len, self.H, self.D)
        k = k.view(seq_len, self.H, self.D)
        v = v.view(seq_len, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        self.cache_K[:, self.P:self.P+seq_len, :] = k
        self.cache_V[:, self.P:self.P+seq_len, :] = v

        q = q.transpose(0, 1)

        scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, self.cache_V)
        output = output.view(seq_len, self.H * self.D)

        return output

class TensorRT_Vanilla(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P  # Add P for cache length

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                # variance = X.pow(2).mean(-1, keepdim=True)
                # X_normed = X * torch.rsqrt(variance)
                # X_normed = self.weight * X_normed

                # QKV projection using separate matrices
                '''
                F.linear(X, W) = X @ W.T
                torch.matmul(X, W) = X @ W
                '''
                # q = F.linear(X_normed, self.W_q)
                # k = F.linear(X_normed, self.W_k)
                # v = F.linear(X_normed, self.W_v)

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)

                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :, :], v], dim=1)
                # cache_K[:, self.P:self.P+seq_len, :] = k
                # cache_V[:, self.P:self.P+seq_len, :] = v

                # scores = torch.matmul(q, cache_K.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                # output = torch.matmul(weights, cache_V)
                # output = output.view(seq_len, self.H*self.D)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V


                # scores = torch.matmul(q, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)
                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)
                # output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)

        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output
    
class TensorRT_Vanilla_GQA(nn.Module):
    def __init__(self, M, N, D, H, N_kv, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P  # Add P for cache length
        self.N_kv = N_kv
        self.H_kv = N_kv // D
        self.group_size = self.H // self.H_kv

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, N_kv, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                self.N_kv = N_kv
                self.H_kv = N_kv // D
                self.group_size = self.H // self.H_kv
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                # variance = X.pow(2).mean(-1, keepdim=True)
                # X_normed = X * torch.rsqrt(variance)
                # X_normed = self.weight * X_normed

                # QKV projection using separate matrices
                '''
                F.linear(X, W) = X @ W.T
                torch.matmul(X, W) = X @ W
                '''
                # q = F.linear(X_normed, self.W_q)
                # k = F.linear(X_normed, self.W_k)
                # v = F.linear(X_normed, self.W_v)

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)

                qkv = torch.matmul(X, self.W_qkv)
                q = qkv[:, :self.N]
                k = qkv[:, self.N:self.N+self.N_kv]
                v = qkv[:, self.N+self.N_kv:]

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H_kv, self.D)
                v = v.view(seq_len, self.H_kv, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :, :], v], dim=1)
                # cache_K[:, self.P:self.P+seq_len, :] = k
                # cache_V[:, self.P:self.P+seq_len, :] = v

                # scores = torch.matmul(q, cache_K.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                # output = torch.matmul(weights, cache_V)
                # output = output.view(seq_len, self.H*self.D)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                k_expanded = k.repeat_interleave(self.group_size, dim=0)
                v_expanded = v.repeat_interleave(self.group_size, dim=0)
                cache_K[:, self.P:self.P+seq_len, :] = k_expanded
                cache_V[:, self.P:self.P+seq_len, :] = v_expanded
                k_cache = cache_K
                v_cache = cache_V


                # scores = torch.matmul(q, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)
                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)
                # output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.N_kv,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)

        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_PreNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                variance = X.pow(2).mean(-1, keepdim=True)
                X_norm = X * torch.rsqrt(variance)

                # q = torch.matmul(X_norm, self.W_q)
                # k = torch.matmul(X_norm, self.W_k)
                # v = torch.matmul(X_norm, self.W_v)
                qkv = torch.matmul(X_norm, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                # scores = torch.matmul(q, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                # output = torch.matmul(weights, v_cache)
                # output = output.view(seq_len, self.H*self.D)
                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)

        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_KeyFormer(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, noise, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.noise = noise.to(device=device, dtype=dtype)
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, noise, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P

                self.noise = noise
                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)
                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)
                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                perturb = (scores+self.noise) / 1.5

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                perturb_exp = torch.exp(perturb)
                perturb_sum = torch.sum(perturb_exp, dim=-1, keepdim=True)
                perturb_out = perturb_exp / perturb_sum

                # weights = F.softmax(scores, dim=-1)
                # perturb_out = F.softmax(perturb, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output, perturb_out

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.noise,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'perturb_out'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        perturb_out = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr(),
            perturb_out.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_NormKeyFormer(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, noise, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.noise = noise.to(device=device, dtype=dtype)
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, noise, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P

                self.noise = noise
                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                variance = X.pow(2).mean(-1, keepdim=True)
                X_norm = X * torch.rsqrt(variance)

                q = torch.matmul(X_norm, self.W_q)
                k = torch.matmul(X_norm, self.W_k)
                v = torch.matmul(X_norm, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                perturb = (scores+self.noise) / 1.5
                weights = F.softmax(scores, dim=-1)
                perturb_out = F.softmax(perturb, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output, perturb_out

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.noise,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'perturb_out'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        perturb_out = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr(),
            perturb_out.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_QKNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)
                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                q_var = q.pow(2).mean(-1, keepdim=True)
                k_var = k.pow(2).mean(-1, keepdim=True)
                q_norm = q * torch.rsqrt(q_var)
                k_norm = k * torch.rsqrt(k_var)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k_norm], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                cache_K[:, self.P:self.P+seq_len, :] = k_norm
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                # scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                # output = torch.matmul(weights, v_cache)
                # output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_NormQKNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                variance = X.pow(2).mean(-1, keepdim=True)
                X_norm = X * torch.rsqrt(variance)

                q = torch.matmul(X_norm, self.W_q)
                k = torch.matmul(X_norm, self.W_k)
                v = torch.matmul(X_norm, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                q_var = q.pow(2).mean(-1, keepdim=True)
                k_var = k.pow(2).mean(-1, keepdim=True)
                q_norm = q * torch.rsqrt(q_var)
                k_norm = k * torch.rsqrt(k_var)

                k_cache = torch.cat([cache_K[:, :self.P, :], k_norm], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
                weights = F.softmax(scores, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_RoCo(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)

                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)
                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                # scores = torch.matmul(q, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                weights_sum = weights.sum(dim=1)
                weights_sqr_sum = weights.pow(2).sum(dim=1)

                # output = torch.matmul(weights, v_cache)
                # output = output.view(seq_len, self.H*self.D)

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output, weights_sum, weights_sqr_sum

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'weights_sum', 'weights_sqr_sum'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        weights_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        weights_sqr_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr(),
            weights_sum.data_ptr(),
            weights_sqr_sum.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class TensorRT_NormRoCo(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P

        # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # RMS Norm
                variance = X.pow(2).mean(-1, keepdim=True)
                X_norm = X * torch.rsqrt(variance)

                q = torch.matmul(X_norm, self.W_q)
                k = torch.matmul(X_norm, self.W_k)
                v = torch.matmul(X_norm, self.W_v)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                weights = F.softmax(scores, dim=-1)

                weights_sum = weights.sum(dim=1)
                weights_sqr_sum = weights.pow(2).sum(dim=1)

                output = torch.matmul(weights, v_cache)
                output = output.view(seq_len, self.H*self.D)

                return output, weights_sum, weights_sqr_sum

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'weights_sum', 'weights_sqr_sum'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        seq_len, H = X.size()

        # Prepare output buffer
        output = torch.empty(seq_len, self.N, dtype=dtype, device=X.device)
        weights_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        weights_sqr_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=X.device)
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            output.data_ptr(),
            weights_sum.data_ptr(),
            weights_sqr_sum.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return output

class FlashInfer_Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # QKV projection
        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_PreNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_KeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.noise = noise

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )
        
        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        weights = F.softmax(scores, dim=-1)
        perturb_out = F.softmax(perturb, dim=-1)

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_NormKeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.noise = noise

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)
        
        # # QKV projection
        q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )
        
        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        weights = F.softmax(scores, dim=-1)
        perturb_out = F.softmax(perturb, dim=-1)

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_QKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # QKV projection
        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        # Transpose all to (H, M, D) for normalization
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        # Normalize Q and K in (H, M, D) format
        q_var = q_t.pow(2).mean(-1, keepdim=True)
        k_var = k_t.pow(2).mean(-1, keepdim=True)
        q_norm_t = q_t * torch.rsqrt(q_var)
        k_norm_t = k_t * torch.rsqrt(k_var)

        # Transpose Q back to (M, H, D) for flashinfer
        q_norm = q_norm_t.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm_t], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v_t], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q_norm,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0  # Important: use 1.0 since Q and K are already normalized
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_NormQKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)
        
        # QKV projection
        q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        # Transpose all to (H, M, D) for normalization
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        # Normalize Q and K in (H, M, D) format
        q_var = q_t.pow(2).mean(-1, keepdim=True)
        k_var = k_t.pow(2).mean(-1, keepdim=True)
        q_norm_t = q_t * torch.rsqrt(q_var)
        k_norm_t = k_t * torch.rsqrt(k_var)

        # Transpose Q back to (M, H, D) for flashinfer
        q_norm = q_norm_t.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm_t], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v_t], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q_norm,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0  # Important: use 1.0 since Q and K are already normalized
        )

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_RoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # QKV projection
        q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )

        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)

        return output.squeeze(0).view(self.M, self.N)

class FlashInfer_NormRoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)
        
        # QKV projection
        q = torch.matmul(X_norm, self.W_q).view(self.M, self.H, self.D)
        k = torch.matmul(X_norm, self.W_k).view(self.M, self.H, self.D)
        v = torch.matmul(X_norm, self.W_v).view(self.M, self.H, self.D)

        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )

        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)

        return output.squeeze(0).view(self.M, self.N)
