import jax
from jax import numpy as jnp
from jax.numpy import concatenate as cat
from minidyn.dyn.body import Body, Inertia
from typing import *


def vec_with1(v):
    return jnp.array([*v, 1]).reshape(4,1)

def vec_rm1(tf):
    return tf[:3]

def vec2world(v, tf):
    Nv= v.shape[0]
    tf1 = jnp.broadcast_to(tf, (Nv, 4, 4))
    tf2 = jax.vmap(vec_with1)(v)#.squeeze()
    tf3 = tf1 @ tf2
    # import pdb;pdb.set_trace()
    v = jax.vmap(vec_rm1)(tf3)
    return v.squeeze()

def vec_normalize(v):
    norm = jnp.linalg.norm(v,axis=1)[:, jnp.newaxis]
    return jnp.where(norm>0, v/norm, v)

def kinetic_energy(inertia, v):
    w = v[:3]#.reshape(1,3)
    y = v[3:]#.reshape(1,3)
    M = inertia.moment
    c = inertia.cross_part
    m = inertia.mass
    T1 = jnp.dot(w, M@w)
    T2 = jnp.dot(y, m*y + 2*jnp.cross(w,c))
    T = (T1 + T2) / 2
    # breakpoint()
    return T
    # return ( (ω,J@ω.T + y@(0.5*(m*y + 2*(jnp.cross(ω,c))).T)) ).reshape(1)
#  ω = angular(twist)
#     v = linear(twist)
#     J = inertia.moment
#     c = inertia.cross_part
#     m = inertia.mass
#     (ω ⋅ (J * ω) + v ⋅ (m * v + 2 * (ω × c))) / 2
def potential_energy(inertia, tf, gravity):
    
    def end(x, e=0.):
        return cat((x, jnp.array((e,)))).reshape(4)
        # return cat((x, jnp.array((e,)))).reshape(4,1)
    
    m = inertia.mass
    g = end(gravity, 0.)
    com = end(inertia.com, 1.)[:, jnp.newaxis]
    com_world = tf @ com

    V = m * jnp.dot(g, com_world)
    # breakpoint()
    return V.squeeze()
    
    # return inertia.mass *  (end(g, 0.).T @ (tf @ end(inertia.com, 1.)))
    # return inertia.mass *  (end(g, 0.).T @ (tf @ end(inertia.com, 1.)))
    #  inertia = spatial_inertia(body)
    # m = inertia.mass
    # m > 0 || return zero(cache_eltype(state))
    # com = transform_to_root(state, body, safe) * center_of_mass(inertia)
    # -m * dot(state.mechanism.gravitational_acceleration, FreeVector3D(com))
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

def quat_norm(quat,eps=1e-9):
    # return quat / (quat @ quat.T)**0.5
    # return vec_normalize(quat)
    # return quat/norm
    norm = jnp.linalg.norm(quat)
    # return jnp.where(norm>eps, quat/norm, quat)# jnp.array([1.,0,0,0]))
    return jnp.where(norm>eps, quat/norm, jnp.array([1.,0,0,0]))


def quat2mat(quat, eps=1e-9):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    Nq = w*w + x*x + y*y + z*z
    def get_mat(w, x, y, z, Nq):
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
    return jnp.where(Nq<eps, jnp.eye(3), get_mat(w, x, y, z, Nq))
    # return  get_mat(w, x, y, z, Nq)
    
def quat_inv(quat):
    return cat((quat[0:1],quat[1:]*-1)) / (quat.T@quat)

def q2tf(q: jnp.array):
    R = quat2mat(quat_norm(q[:4]))
    T = q[4:].reshape(3, 1)
    B = jnp.array((0., 0., 0., 1.)).reshape(1, 4)
    # breakpoint()
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

def stack_attr(pytrees, attr):
    return jax.tree_util.tree_map( lambda *values: 
        jnp.stack(values, axis=0), *[getattr(t, attr) for t in pytrees])
def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaFn.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [jax.tree_util.unflatten(l) for l in new_leaves]
    return new_trees

# def breakpoint_fn(x):
#     cond = jnp.any(x)
#     def true_fn(x):
#         pass
#     def false_fn(x):
#         jax.debug.breakpoint()
#     jax.lax.cond(cond, true_fn, false_fn, x)
# breakpoint_fn(did_collides)# def breakpoint_fn(x):
#     cond = jnp.any(x)
#     def true_fn(x):
#         pass
#     def false_fn(x):
#         jax.debug.breakpoint()
#     jax.lax.cond(cond, true_fn, false_fn, x)
# breakpoint_fn(did_collides)