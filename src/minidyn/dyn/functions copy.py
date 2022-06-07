import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
from minidyn.dyn.body import Body
import networkx
from minidyn.dyn.spatial import Inertia
from typing import *

from jax import vmap, tree_multimap,lax

def kinetic_energy(inertia, v):
    ω = v[:3].reshape(1,3)
    s = v[3:].reshape(1,3)
    J = inertia.moment
    c = inertia.cross_part
    m = inertia.mass
    return ((ω @ (J @ ω.T) + s @ (m * s + 2 * (jnp.cross(ω,c))).T) / 2).reshape(1)

def potential_energy(inertia, tf, g):
    def end(x, e=0.):
        return cat((x, jnp.array((e,)))).reshape(4,1)
    return inertia.mass *  (end(g, 0.).T @ (tf @ end(inertia.com, 1.)))
    
def inertia_to_world(inertia, tf):
    J = inertia.moment
    mc = inertia.cross_part.reshape(3,1)
    m = inertia.mass
    R = tf[:3,:3]
    p = tf[4,:3].reshape(1,3)

    Rmc = R @ mc
    mp = m * p
    mcnew = (Rmc + mp.T).reshape(3)
    X = Rmc @ p
    Y = X + X.T + mp @ p.T
    Jnew = R @ J @ R.T - Y + jnp.trace(Y) * jnp.eye(3)

    return Inertia(m, Jnew, mcnew)

def quat_norm(quat):
    quat = quat
    return quat / (quat @ quat.T)**0.5

def quat2mat(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    Nq = w*w + x*x + y*y + z*z
    # import pdb;pdb.set_trace()
    # if (Nq < 1e-9):
    #     return jnp.eye(3)
    # else:
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return jnp.array(
        [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
        [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
        [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def q2tf(q: jnp.array):
    R = quat2mat(quat_norm(q[:4]))
    T = q[4:].reshape(3, 1)
    B = jnp.array((0., 0., 0., 1.)).reshape(1, 4)
    return cat((cat((R, T),1), B),0)

def qqd2v(q, qdot):
    quat = quat_norm(q[0:4])
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    vjac_angvel = 2 * jnp.array([
        [-x, w, z, -y],
        [-y, -z, w, x],
        [-z, y, -x, w],
        ])
    quatdot = quat_norm(qdot[0:4])
    angvel = vjac_angvel @ quatdot
    posdot = qdot[4:]
    quat_inv = cat((quat[0:1],quat[1:]*-1)) / (quat@quat)
    linvel = quat2mat(quat_inv) @ posdot
    return cat((angvel, linvel)) # 7-dim -> 6-dim

def qv2qd(q, v):
    quat = quat_norm(q[0:4])
    angvel = v[:3]
    linvel = v[3:]
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    vjac_quatdot =  jnp.array([
                        [-x, -y, -z],
                        [w, -z, y],
                        [z,  w, -x],
                        [-y,  x,  w]]) / 2
    quatdot = vjac_quatdot @ angvel
    transdot = quat2mat(quat) @ linvel
    return jnp.hstack((quatdot, transdot)) # 6-dim -> 7-dim


def se3_commutator(xω, xv, yω, yv):
    ang = torch.cross(xω, yω)
    lin = torch.cross(xω, yv) + torch.cross(xv, yω)
    return ang, lin

def vjac_quatdot(quat):
    w, x, y, z = quat
    return torch.Tensor([
        [-x, -y, -z],
        [w, -z, y],
        [z,  w, -x],
        [-y,  x,  w]], device=quat.device) / 2

def vjac_angvel(quat): # quat_dot.T * 4 
    w, x, y, z = quat
    return 2 * torch.Tensor([
        [-x, w, z, -y],
        [-y, -z, w, x],
        [-z, y, -x, w],
        ], device=quat.device)

def quatquatdot2angvel(quat, quatdot):
    return vjac_angvel(quat) @ quatdot

def quatangvel2quatdot(quat, angvel):
    return vjac_quatdot(quat) @ angvel

def quat_inv(quat):
    return torch.cat((quat[0],quat[1:]*-1)) / (quat@quat)