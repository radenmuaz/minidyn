# import jax
# from jax import numpy as jnp, random
# from dataclasses import dataclass
'''
@dataclass
class Joint: 
    prev_frame: Frame
    next_frame: Frame
    tf_pred: Transform3D
    tf_succ: Transform3D
    qlim: jnp.array
    vlim: jnp.array
    elim: jnp.array
    nq: int
    nv: int
    
    # def __post_init__(self):
    #     self.id = 0

@dataclass
class FreeJoint(Joint):
    qlim: jnp.array = jnp.array((-jnp.inf, jnp.inf))
    vlim: jnp.array = jnp.array((-jnp.inf, jnp.inf))
    elim: jnp.array = jnp.array((-jnp.inf, jnp.inf))
    nq: int = 7
    nv: int = 6   
    def quat_norm(self, quat):
        return quat / (quat @ quat)**0.5
    
    def quat2mat(self, quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        Nq = w*w + x*x + y*y + z*z
        if Nq < 1e-9:
            return jnp.eye(3)
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
    
    def get_tf(self, q: jnp.array, next_frame: Frame, prev_frame: Frame):
        return Transform3D(next_frame, prev_frame, self.quat2mat(self.quat_norm(q[:4])), q[4:])
    
    def get_bias_accel(self, next_frame: Frame, prev_frame: Frame):
        return Motion(jnp.zeros(3,), jnp.zeros(3,), next_frame, prev_frame, next_frame)

    def get_motion_subspace(self, q, frame_next: Frame, frame_prev:Frame):
        assert len(q) == 7
        ang = jnp.hstack((jnp.ones(3,3), jnp.zeros(3,3)),
                             dim=1)
        lin = jnp.hstack((jnp.zeros(3,3), jnp.ones(3,3)),
                            dim=1)
        return Motion(ang=ang, lin=lin, frame=frame_next, body=frame_next, base=frame_prev)
    
    def get_constraint_subspace(self, tf: Transform3D):
        zeros = jnp.zeros(3)
        return Motion(zeros, zeros, tf.prev)

    def get_bias_accel(self, frame_next: Frame, frame_prev: Frame):
        return Motion(jnp.zeros(1), frame=frame_next, body=frame_next, base=frame_prev)

    def qqdot2v(self, q, qdot):
        quat = self.quat_norm(q[0:4])
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        vjac_angvel = 2 * jnp.array([
            [-x, w, z, -y],
            [-y, -z, w, x],
            [-z, y, -x, w],
            ])
        quatdot = self.quat_norm(qdot[0:4])
        angvel = vjac_angvel @ quatdot
        posdot = qdot[4:]
        quat_inv = jnp.hstack((quat[0],quat[1:]*-1)) / (quat@quat)
        linvel = quat_inv @ posdot
        return jnp.hstack((angvel, linvel)) # 7-dim -> 6-dim
    
    def qv2qdot(self, q, v):
        quat = self.quat_norm(q[0:4])
        angvel = v[:3]
        linvel = v[3:]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        vjac_quatdot =  jnp.array([
                            [-x, -y, -z],
                            [w, -z, y],
                            [z,  w, -x],
                            [-y,  x,  w]]) / 2
        quatdot = vjac_quatdot @ angvel
        transdot = quat @ linvel
'''