# https://albert.growi.cloud/627ceb255286c3b02691384d
import torch
import torch.nn.functional as F
def hat(phi):
    phi_x = phi[..., 0]
    phi_y = phi[..., 1]
    phi_z = phi[..., 2]
    zeros = torch.zeros_like(phi_x)

    phi_hat = torch.stack([
        torch.stack([zeros, -phi_z,  phi_y], dim=-1),
        torch.stack([phi_z,  zeros, -phi_x], dim=-1),
        torch.stack([-phi_y,  phi_x,  zeros], dim=-1)
    ], dim=-2)
    return phi_hat


def ExpSO3(phi, eps=1e-4):
    theta = torch.norm(phi, dim=-1)
    phi_hat = hat(phi)
    E = torch.eye(3, device=phi.device)
    coef1 = torch.zeros_like(theta)
    coef2 = torch.zeros_like(theta)

    ind = theta < eps

    # strict
    _theta = theta[~ind]
    coef1[~ind] = torch.sin(_theta) / _theta
    coef2[~ind] = (1 - torch.cos(_theta)) / _theta**2

    # approximate
    _theta = theta[ind]
    _theta2 = _theta**2
    _theta4 = _theta**4
    coef1[ind] = 1 - _theta2/6 + _theta4/120
    coef2[ind] = .5 - _theta2/24 + _theta4/720

    coef1 = coef1[..., None, None]
    coef2 = coef2[..., None, None]
    return E + coef1 * phi_hat + coef2 * phi_hat @ phi_hat

def drct_rotation(drct_src, drct_dst):
    #Size: B,P,3
    #R @ src.unsqueeze(-1) = dst

    B,P = drct_dst.shape[0], drct_dst.shape[1]
    drct_n = drct_dst / torch.norm(drct_dst, dim=-1, keepdim=True)
    drct_o = drct_src / torch.norm(drct_src, dim=-1, keepdim=True)

    inner = torch.clamp((drct_n * drct_o).sum(-1), min=-1 + 1e-4, max=1 - 1e-4)
    theta = torch.acos(inner)#B,P
    cross_drct = torch.cross(drct_o, drct_n, dim=-1)
    cross_drct_n = cross_drct / torch.norm(cross_drct,dim=-1,keepdim=True)#B,P,3

    pos_r = ExpSO3(cross_drct_n * theta.unsqueeze(-1))
    pos_dummy = torch.einsum('bpij, bpjk -> bpik', pos_r, drct_o.unsqueeze(-1))
    pos_distance = torch.norm(pos_dummy.squeeze(-1) - drct_n, dim=-1)
    neg_r = ExpSO3(-cross_drct_n * theta.unsqueeze(-1))
    neg_dummy = torch.einsum('bpij, bpjk -> bpik', neg_r, drct_o.unsqueeze(-1))
    neg_distance = torch.norm(neg_dummy.squeeze(-1) - drct_n, dim=-1)

    ret = []
    for b in range(B):
        for p in range(P):
            if neg_distance[b,p] < pos_distance[b,p]:
                ret.append(neg_r[b,p])
            else:
                ret.append(pos_r[b,p])
    ret = torch.stack(ret, dim=0).reshape(B,P,3,3)
    return ret

#from https://memo.sugyan.com/entry/2022/09/09/230645
def slerp(
    t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995
    ) -> torch.Tensor:
    '''
    return (spherical) linear interpolation between v1 & v2
    w0 = 1-t, w1 = t
    '''
    shape = v0.shape
    v0, v1 = v0.reshape(-1,3), v1.reshape(-1,3)

    u0 = v0 / torch.norm(v0, dim=-1,keepdim=True)
    u1 = v1 / torch.norm(v1, dim=-1,keepdim=True)
    dot = (u0 * u1).sum(-1)

    ret_linear = (1.0 - t) * v0 + t * v1
    omega = torch.acos(dot).unsqueeze(-1)
    ret_slerp = (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()
    
    linear_idx = (dot.abs() > DOT_THRESHOLD) + torch.isnan(ret_slerp[:,0])
    linear_idx = F.one_hot(linear_idx.long(), num_classes=2)
    ret = torch.stack([ret_slerp, ret_linear], dim=1)
    ret = (ret * linear_idx.unsqueeze(-1)).sum(1).reshape(*shape)
    ret_n = ret / torch.norm(ret, dim=-1, keepdim=True)
    return ret_n