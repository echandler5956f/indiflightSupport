# Simplified multirotor flight dynamics and sensor models
#
# Copyright 2024 Till Blaha (Delft University of Technology)
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R
from scipy.constants import g as GRAVITY

from .helpers import (
    cross,
    quatRotate,
    quaternionDerivative,
    angularRateDerivative,
    rotatingMassTorques,
    motorModel
    )


@njit(cache=True)
def _quat_rotate(q, vx, vy, vz):
    q0 = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        vx + q0 * tx + (qy * tz - qz * ty),
        vy + q0 * ty + (qz * tx - qx * tz),
        vz + q0 * tz + (qx * ty - qy * tx),
    )


@njit(cache=True)
def _matvec3(A, x0, x1, x2):
    return (
        A[0, 0] * x0 + A[0, 1] * x1 + A[0, 2] * x2,
        A[1, 0] * x0 + A[1, 1] * x1 + A[1, 2] * x2,
        A[2, 0] * x0 + A[2, 1] * x1 + A[2, 2] * x2,
    )


@njit(cache=True)
def _multirotor_tick_fast(
    inputs,
    rotor_velocity,
    rotor_r,
    rotor_axis,
    rotor_dir,
    rotor_k,
    rotor_cm,
    rotor_tau,
    rotor_wmax,
    rotor_kesc,
    rotor_izz,
    m,
    I,
    Iinv,
    xI,
    vI,
    fspB,
    q,
    wDotB,
    wB,
    FthrowI,
    MthrowB,
    throw_time,
    throw_duration,
    dt,
):
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0
    Mx = 0.0
    My = 0.0
    Mz = 0.0

    for i in range(rotor_velocity.shape[0]):
        u = inputs[i]
        wc = rotor_wmax[i] * np.sqrt(rotor_kesc[i] * u * u + (1.0 - rotor_kesc[i]) * u)
        w_dot = (wc - rotor_velocity[i]) / rotor_tau[i]
        w = rotor_velocity[i] + dt * w_dot
        rotor_velocity[i] = w

        ax = rotor_axis[i, 0]
        ay = rotor_axis[i, 1]
        az = rotor_axis[i, 2]
        thrust = rotor_k[i] * w * w
        f0 = ax * thrust
        f1 = ay * thrust
        f2 = az * thrust

        Fx += f0
        Fy += f1
        Fz += f2

        rx = rotor_r[i, 0]
        ry = rotor_r[i, 1]
        rz = rotor_r[i, 2]
        mx = ry * f2 - rz * f1
        my = rz * f0 - rx * f2
        mz = rx * f1 - ry * f0

        drag_scale = rotor_dir[i] * rotor_cm[i]
        mx -= drag_scale * f0
        my -= drag_scale * f1
        mz -= drag_scale * f2

        sx = ax * rotor_dir[i]
        sy = ay * rotor_dir[i]
        sz = az * rotor_dir[i]
        lx = rotor_izz[i] * w * sx
        ly = rotor_izz[i] * w * sy
        lz = rotor_izz[i] * w * sz
        dldtx = rotor_izz[i] * w_dot * sx
        dldty = rotor_izz[i] * w_dot * sy
        dldtz = rotor_izz[i] * w_dot * sz
        cx = wB[1] * lz - wB[2] * ly
        cy = wB[2] * lx - wB[0] * lz
        cz = wB[0] * ly - wB[1] * lx
        mx -= dldtx + cx
        my -= dldty + cy
        mz -= dldtz + cz

        Mx += mx
        My += my
        Mz += mz

    qinv0 = -q[0]
    qinv = np.empty(4, dtype=np.float32)
    qinv[0] = qinv0
    qinv[1] = q[1]
    qinv[2] = q[2]
    qinv[3] = q[3]

    if xI[2] > 0.0:
        frx, fry, frz = _quat_rotate(qinv, 0.0, 0.0, -xI[2])
        Fx += 1000.0 * m * frx
        Fy += 1000.0 * m * fry
        Fz += 1000.0 * m * frz

        damping = 100.0 if vI[2] > 0.0 else 1.0
        fvx, fvy, fvz = _quat_rotate(qinv, -vI[0], -vI[1], -vI[2])
        Fx += damping * m * fvx
        Fy += damping * m * fvy
        Fz += damping * m * fvz

        sign_q = 1.0 if qinv0 >= 0.0 else -1.0
        ix, iy, iz = _matvec3(I, sign_q * qinv[1], sign_q * qinv[2], sign_q * qinv[3])
        Mx += 1000.0 * ix
        My += 1000.0 * iy
        Mz += 1000.0 * iz

        dx, dy, dz = _matvec3(I, -wB[0], -wB[1], -wB[2])
        Mx += 100.0 * dx
        My += 100.0 * dy
        Mz += 100.0 * dz

    throw_time += dt
    if throw_time > 0.0 and throw_time <= throw_duration:
        tfx, tfy, tfz = _quat_rotate(qinv, FthrowI[0], FthrowI[1], FthrowI[2])
        Fx += tfx
        Fy += tfy
        Fz += tfz
        Mx += MthrowB[0]
        My += MthrowB[1]
        Mz += MthrowB[2]

    iw0, iw1, iw2 = _matvec3(I, wB[0], wB[1], wB[2])
    cx = wB[1] * iw2 - wB[2] * iw1
    cy = wB[2] * iw0 - wB[0] * iw2
    cz = wB[0] * iw1 - wB[1] * iw0
    rhs0 = Mx - cx
    rhs1 = My - cy
    rhs2 = Mz - cz
    wdot0, wdot1, wdot2 = _matvec3(Iinv, rhs0, rhs1, rhs2)
    wDotB[0] = wdot0
    wDotB[1] = wdot1
    wDotB[2] = wdot2

    wx = 0.5 * wB[0]
    wy = 0.5 * wB[1]
    wz = 0.5 * wB[2]
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    qdot0 = -wx * qx - wy * qy - wz * qz
    qdot1 = wx * qw + wz * qy - wy * qz
    qdot2 = wy * qw - wz * qx + wx * qz
    qdot3 = wz * qw + wy * qx - wx * qy

    fspB[0] = Fx / m
    fspB[1] = Fy / m
    fspB[2] = Fz / m

    vdot0, vdot1, vdot2 = _quat_rotate(q, fspB[0], fspB[1], fspB[2])
    vdot2 += GRAVITY

    xdot0 = vI[0]
    xdot1 = vI[1]
    xdot2 = vI[2]

    wB[0] += dt * wdot0
    wB[1] += dt * wdot1
    wB[2] += dt * wdot2

    q[0] += dt * qdot0
    q[1] += dt * qdot1
    q[2] += dt * qdot2
    q[3] += dt * qdot3
    q_norm_inv = 1.0 / np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q[0] *= q_norm_inv
    q[1] *= q_norm_inv
    q[2] *= q_norm_inv
    q[3] *= q_norm_inv

    vI[0] += dt * vdot0
    vI[1] += dt * vdot1
    vI[2] += dt * vdot2
    xI[0] += dt * xdot0
    xI[1] += dt * xdot1
    xI[2] += dt * xdot2

    return throw_time


@njit(cache=True)
def _imu_update_no_noise(uav_fspB, uav_wDotB, uav_wB, r, qInv, acc, gyro):
    cx1 = uav_wDotB[1] * r[2] - uav_wDotB[2] * r[1]
    cy1 = uav_wDotB[2] * r[0] - uav_wDotB[0] * r[2]
    cz1 = uav_wDotB[0] * r[1] - uav_wDotB[1] * r[0]

    wxr0 = uav_wB[1] * r[2] - uav_wB[2] * r[1]
    wxr1 = uav_wB[2] * r[0] - uav_wB[0] * r[2]
    wxr2 = uav_wB[0] * r[1] - uav_wB[1] * r[0]
    cx2 = uav_wB[1] * wxr2 - uav_wB[2] * wxr1
    cy2 = uav_wB[2] * wxr0 - uav_wB[0] * wxr2
    cz2 = uav_wB[0] * wxr1 - uav_wB[1] * wxr0

    ax, ay, az = _quat_rotate(
        qInv,
        uav_fspB[0] + cx1 + cx2,
        uav_fspB[1] + cy1 + cy2,
        uav_fspB[2] + cz1 + cz2,
    )
    gx, gy, gz = _quat_rotate(qInv, uav_wB[0], uav_wB[1], uav_wB[2])

    acc[0] = ax
    acc[1] = ay
    acc[2] = az
    gyro[0] = gx
    gyro[1] = gy
    gyro[2] = gz

class Rotor:
    def __init__(self, r=[0., 0., 0.], axis=[0., 0., -1.], wmax=4900., Tmax=4.5, kESC=0.5, cm=0.01, tau=0.02, Izz=1e-6, dir='rh'):
        self.r = np.asarray(r, dtype=np.float32)
        self.axis = np.asarray(axis, dtype=np.float32)
        self.axis /= np.linalg.norm(self.axis)
        self.wmax = wmax
        self.Tmax = Tmax
        self.kESC = kESC
        self.k = Tmax / self.wmax / self.wmax
        self.cm = cm
        self.tau = tau
        self.Izz = Izz
        if dir in ['rh', 'lh']:
            self.dir = -1. if dir=='lh' else +1.
        else:
            raise ValueError("dir must be one of 'rh', or 'lh'!")

        self.F = np.array([0., 0., 0.], dtype=np.float32)
        self.M = np.array([0., 0., 0.], dtype=np.float32)
        self.w = 0.

    def step(self, u, Omega, dt):
        wDot = motorModel( u, self.kESC, self.wmax, self.w, self.tau )
        self.w += dt * wDot
        self.F = self.axis * self.k * self.w*self.w
        self.M = cross(self.r, self.F)
        self.M -= self.dir * self.cm * self.F
        self.M -= rotatingMassTorques(self.Izz, self.axis*self.dir, self.w, wDot, Omega)

class MultiRotor:
    def __init__(self):
        self.rotors = []
        self.m = 1.
        self.I = np.eye(3, dtype=np.float32)
        self.Iinv = np.eye(3, dtype=np.float32)
        self.xI = np.array([0., 0., 0.], dtype=np.float32)
        self.vI = np.array([0., 0., 0.], dtype=np.float32)
        self.fspB = np.array([0., 0., 0.], dtype=np.float32)
        self.q = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.wDotB = np.array([0., 0., 0.], dtype=np.float32)
        self.wB = np.array([0., 0., 0.], dtype=np.float32)

        self.throw_time = +np.inf
        self.throw_duration = 0.
        self.FthrowI = np.array([0., 0., 0.], dtype=np.float32)
        self.MthrowB = np.array([0., 0., 0.], dtype=np.float32)
        self.use_fast_tick = True

    def __repr__(self):
        qWrong = np.zeros_like(self.q)
        qWrong[3] = self.q[0]
        qWrong[:3] = self.q[1:]
        rot = R.from_quat(qWrong)
        eulers = rot.as_euler('ZYX', degrees=True)
        return f"MultiRotor( n={len(self.rotors)}, x={self.xI}m, v={self.vI}m/s, roll={eulers[2]}deg, pitch={eulers[1]}deg, yaw={eulers[0]}deg )"

    def throw(self, height=3.5, acc=45., wB=[0., 0., 0.], vHorz=[0., 0.], at_time=0.):
        force = self.m * ( acc + GRAVITY )

        # solve duration:
        # 
        # height = height after powered throw (sT) + altitude gained during coasting (sC)
        # sT = 0.5*a*t**2
        # vT = a*t
        # sC = 0.5*vT**2 / g,  becayse 0.5*vT**2 = g*sC
        # 
        # then, solve  height == sT + sC  for time
        self.throw_time = -at_time
        self.throw_duration = np.sqrt( 2. * height / (acc * (1. + acc / GRAVITY)) )

        self.FthrowI[:2] = self.m * np.asarray(vHorz) / self.throw_duration
        self.FthrowI[2] = -force
        self.MthrowB[:] = self.I @ ( wB / self.throw_duration )

    def setInertia(self, m, I):
        self.m = m
        self.I = I.astype(np.float32)
        self.Iinv = np.linalg.inv(I).astype(np.float32)

    def addRotor(self, rotor):
        self.rotors.append(rotor)
        self.n = len(self.rotors)
        self.rotorVelocity = np.zeros(self.n, np.float32)
        self.inputs = np.zeros(self.n, np.float32)
        self._refreshRotorCache()
        self._warmFastTick()

    def _refreshRotorCache(self):
        self._rotor_r = np.ascontiguousarray([rotor.r for rotor in self.rotors], dtype=np.float32)
        self._rotor_axis = np.ascontiguousarray([rotor.axis for rotor in self.rotors], dtype=np.float32)
        self._rotor_dir = np.ascontiguousarray([rotor.dir for rotor in self.rotors], dtype=np.float32)
        self._rotor_k = np.ascontiguousarray([rotor.k for rotor in self.rotors], dtype=np.float32)
        self._rotor_cm = np.ascontiguousarray([rotor.cm for rotor in self.rotors], dtype=np.float32)
        self._rotor_tau = np.ascontiguousarray([rotor.tau for rotor in self.rotors], dtype=np.float32)
        self._rotor_wmax = np.ascontiguousarray([rotor.wmax for rotor in self.rotors], dtype=np.float32)
        self._rotor_kesc = np.ascontiguousarray([rotor.kESC for rotor in self.rotors], dtype=np.float32)
        self._rotor_izz = np.ascontiguousarray([rotor.Izz for rotor in self.rotors], dtype=np.float32)

    def _warmFastTick(self):
        if not self.use_fast_tick or not self.rotors:
            return
        self.throw_time = _multirotor_tick_fast(
            self.inputs,
            self.rotorVelocity,
            self._rotor_r,
            self._rotor_axis,
            self._rotor_dir,
            self._rotor_k,
            self._rotor_cm,
            self._rotor_tau,
            self._rotor_wmax,
            self._rotor_kesc,
            self._rotor_izz,
            self.m,
            self.I,
            self.Iinv,
            self.xI,
            self.vI,
            self.fspB,
            self.q,
            self.wDotB,
            self.wB,
            self.FthrowI,
            self.MthrowB,
            self.throw_time,
            self.throw_duration,
            0.0,
        )

    def setPose(self, x=[0., 0., 0.], q=[1., 0., 0., 0.]):
        self.xI[:] = np.asarray(x, dtype=np.float32)
        self.q[:] = np.asarray(q, dtype=np.float32)

    def setTwist(self, v=[0., 0., 0.], w=[0., 0., 0.]):
        self.vI[:] = np.asarray(v, dtype=np.float32)
        self.wB[:] = np.asarray(w, dtype=np.float32)

    def setExternalForceInInertialFrame(self, F):
        self.FthrowI[:] = F

    def setExternalMomentInBodyFrame(self, M):
        self.MthrowB[:] = M

    def calculateG1G2(self):
        # 2024-02-25 slightly nicer formulation for online learning (G2 not scaled with Tmax)
        # 
        #  let O = (Fx Fy Fz Mx My Mz)
        # idea: DeltaO = B1 * DeltaT  +  B2 * DeltaWdot
        # 
        # where B1 holds information about thrust axes and motor locations
        # and   B2 holds information about thrust axes and propeller inertia
        # 
        # using w = sqrt(T/k), first-order dynamics wdot = (w - w0)/tau and taylor 
        # expansion of the square root results in:
        # 
        #   DeltaO = B1 DeltaT  +  B2 / (2*w0*tau*k) * (DeltaT - DeltaTprev)
        #
        # Introduce the normalized unitless control U = T / Tmax
        #
        #   DeltaO = B1 Tmax DeltaU                +  B2 * Tmax / (2*tau*k*w0) * (DeltaU - DeltaUprev)
        #   DeltaO = B1 * k * omegaMax^2 * DeltaU  +  B2 * omegaMax^2 / (2*tau*w0) * (DeltaU - DeltaUprev)
        #
        # Introduce specific generalized forces A = (fx fy fz taux tauy tauz) with 
        # units (N/kg N/kg N/kg Nm/(kgm^2) Nm/(kgm^2) Nm/(kgm^2)) and
        #
        #   DeltaA = G1 DeltaU  +  G2 * omegaMax^2 / (2*tau*w0) * (DeltaU - DeltaUprev)
        #      where  G1   == (Minv B1) * k * omegaMax^2  , where (Minv B1 * k) can be learned online and then scaled with omegaMax^2 which is separetely learned online
        #        or   G1   == (Minv B1) * Tmax            , which seems more accurate, if available
        #      and    G2   == (Minv B2)                   , which can be learned online
        #      and    Minv == inv(diag(m,m,m,Ixx,Iyy,Izz)), called generalized mass matrix
        # 
        # this can later be inverted to compute DeltaU by solving:
        #
        #   DeltaA + G2n / w0 DeltaU_prev = ( G1 + G2n / w0 )  DeltaU
        #      where G2n = G2 * omegaMax^2 / (2*tau)
        #
        # or, assuming wdot feedback is available
        #
        #   DeltaA + G2 * wdot_prev = ( G1 + G2n / w0 ) DeltaU
        ##################
        N = len(self.rotors)

        B1 = np.zeros((6, N))
        B2 = np.zeros((6, N))
        for i, rotor in enumerate(self.rotors):
            # force contribution from thrust
            B1[:3, i] = rotor.axis

            # moment contribution from thrust
            # and moment contribution from rotor drag
            B1[3:, i] = np.cross(rotor.r, rotor.axis) \
                        -rotor.dir * rotor.cm * rotor.axis

            B1[:, i] *= rotor.Tmax

            # moment contribution from spinup
            B2[3:, i] = -rotor.dir * rotor.axis * rotor.Izz

        M = np.zeros((6,6))
        M[:3, :3] = self.m * np.eye(3)
        M[3:, 3:] = np.diag(np.diag(self.I)) # remove offdiagonal elements
        #M[3:, 3:] = self._I # isnt this more accurate?

        G1 = np.linalg.solve(M, B1)
        G2 = np.linalg.solve(M, B2)
        G2_scaler = np.array([0.5 * r.wmax**2 / (0.5*r.tau) for r in self.rotors])
        return G1, G2, G2_scaler

    def checkHover(self):
        G1, _, _ = self.calculateG1G2()
        Qr, Rr = np.linalg.qr(G1[3:, :].T, 'complete')
        if (len(self.rotors) < 4) or (np.abs(np.diag(Rr)) < 1e-3).any():
            print(f"\nWARNING: generated craft has no control over some rotation axis or axes.")
            return False
        else:
            Nr = Qr[:, 3:] # rotational nullspace
            A = Nr.T @ G1[:3, :].T @ G1[:3, :] @ Nr
            v, V = np.linalg.eig(A)
            # calculate most effeicient hover allocation with 1.1 thrust to weight margin
            ustar = ( 1.1 * 9.81 / np.sqrt(max(v)) ) * (Nr @ V[:, np.argmax(v)])
            if not ( ((ustar >= 0.) & (ustar <= 1.)).all() or ((ustar >= -1.) & (ustar <= 0.)).all()):
                print(f"\nWARNING: generated craft does not have enough thrust-to-weight to hover without rotation. Double check rotation directions")
                return False

        return True

    def tick(self, dt):
        if self.use_fast_tick and len(self.rotors) == len(self.inputs):
            self.throw_time = _multirotor_tick_fast(
                self.inputs,
                self.rotorVelocity,
                self._rotor_r,
                self._rotor_axis,
                self._rotor_dir,
                self._rotor_k,
                self._rotor_cm,
                self._rotor_tau,
                self._rotor_wmax,
                self._rotor_kesc,
                self._rotor_izz,
                self.m,
                self.I,
                self.Iinv,
                self.xI,
                self.vI,
                self.fspB,
                self.q,
                self.wDotB,
                self.wB,
                self.FthrowI,
                self.MthrowB,
                self.throw_time,
                self.throw_duration,
                dt,
            )
            for i, rotor in enumerate(self.rotors):
                rotor.w = float(self.rotorVelocity[i])
            return

        F = np.zeros(3, dtype=np.float32)
        M = np.zeros(3, dtype=np.float32)
        for i, (u, rotor) in enumerate(zip(self.inputs, self.rotors)):
            rotor.step(u, self.wB, dt)
            self.rotorVelocity[i] = rotor.w
            F += rotor.F
            M += rotor.M

        qInv = self.q.copy()
        qInv[0] *= -1.
        if self.xI[2] > 0.:
            # handle ground contact
            down = self.vI[2] > 0.
            F += (1000 if down else 1000)  * self.m * quatRotate( qInv, np.array([0., 0., -1.], dtype=np.float32) * self.xI )
            F += (100  if down else 1) * self.m * quatRotate( qInv, -self.vI )
            M += 1000 * self.I @ ( np.sign(qInv[0]) * qInv[1:] )
            M += 100 * self.I @ -self.wB

        # throw timekeeping and add external force and moment
        self.throw_time += dt
        if self.throw_time > 0. and self.throw_time <= self.throw_duration:
            F += quatRotate( qInv, self.FthrowI )
            M += self.MthrowB

        self.wDotB[:] = angularRateDerivative( self.wB, M, self.I, self.Iinv )
        qDot = quaternionDerivative( self.q, self.wB )
        self.fspB[:] = F / self.m
        vDot = quatRotate( self.q, self.fspB )  +  np.array([0., 0., GRAVITY])
        xDot = self.vI

        self.wB += dt * self.wDotB
        self.q += dt * qDot
        self.q /= np.linalg.norm(self.q)
        self.vI += dt * vDot
        self.xI += dt * xDot

class IMU:
    def __init__(self, uav, r=[0., 0., 0.], qBody=[1., 0., 0., 0.], accBias=[0., 0., 0.], accStd=0.0, gyroBias=[0., 0., 0.], gyroStd=0.0):
        self.uav = uav

        self.r = np.asarray(r, dtype=np.float32)
        self.qInv = np.asarray(qBody, dtype=np.float32)
        self.qInv[0] *= -1.

        self.acc = np.zeros(3, dtype=np.float32)
        self.accBias = np.asarray(accBias, dtype=np.float32)
        self.accStd = accStd

        self.gyro = np.zeros(3, dtype=np.float32)
        self.gyroBias = np.asarray(gyroBias, dtype=np.float32)
        self.gyroStd = gyroStd
        if self.accStd == 0.0 and self.gyroStd == 0.0:
            _imu_update_no_noise(
                self.uav.fspB,
                self.uav.wDotB,
                self.uav.wB,
                self.r,
                self.qInv,
                self.acc,
                self.gyro,
            )

    def update(self):
        if self.accStd == 0.0 and self.gyroStd == 0.0:
            _imu_update_no_noise(
                self.uav.fspB,
                self.uav.wDotB,
                self.uav.wB,
                self.r,
                self.qInv,
                self.acc,
                self.gyro,
            )
            return

        accAtImu = self.uav.fspB + cross(self.uav.wDotB, self.r) + cross(self.uav.wB, cross(self.uav.wB, self.r))
        self.acc[:] = quatRotate(self.qInv, accAtImu) + np.random.normal(self.accBias, self.accStd)
        self.gyro[:] = quatRotate(self.qInv, self.uav.wB) + np.random.normal(self.gyroBias, self.gyroStd)
