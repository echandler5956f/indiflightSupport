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
def _multirotor_derivative(
    s, inputs, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm, rotor_tau, rotor_wmax, rotor_kesc, rotor_izz,
    m, I, Iinv, FextI, MextB, external, drag_linear, drag_quadratic, gravity, fspB, wDotB,
):
    # The derivative of s = [rotor speeds (n), wB (3), q (4, scalar first), vI (3), xI (3)] under the rotors, ground
    # contact, the external wrench while it acts and drag, with the motor commands held; writes the specific force and
    # the angular acceleration at s.
    n = rotor_r.shape[0]
    ds = np.zeros_like(s)
    wb0 = s[n]
    wb1 = s[n + 1]
    wb2 = s[n + 2]
    qn = np.sqrt(s[n + 3] ** 2 + s[n + 4] ** 2 + s[n + 5] ** 2 + s[n + 6] ** 2)
    q = np.empty(4)
    q[0] = s[n + 3] / qn
    q[1] = s[n + 4] / qn
    q[2] = s[n + 5] / qn
    q[3] = s[n + 6] / qn
    v0 = s[n + 7]
    v1 = s[n + 8]
    v2 = s[n + 9]
    z = s[n + 12]
    Fx = 0.0
    Fy = 0.0
    Fz = 0.0
    Mx = 0.0
    My = 0.0
    Mz = 0.0
    for i in range(n):
        u = np.float64(inputs[i])
        wc = rotor_wmax[i] * np.sqrt(rotor_kesc[i] * u * u + (1.0 - rotor_kesc[i]) * u)
        w = s[i]
        w_dot = (wc - w) / rotor_tau[i]
        ds[i] = w_dot
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
        mx -= rotor_izz[i] * w_dot * sx + (wb1 * lz - wb2 * ly)
        my -= rotor_izz[i] * w_dot * sy + (wb2 * lx - wb0 * lz)
        mz -= rotor_izz[i] * w_dot * sz + (wb0 * ly - wb1 * lx)
        Mx += mx
        My += my
        Mz += mz
    qinv = np.empty(4)
    qinv[0] = -q[0]
    qinv[1] = q[1]
    qinv[2] = q[2]
    qinv[3] = q[3]
    if z > 0.0:
        frx, fry, frz = _quat_rotate(qinv, 0.0, 0.0, -z)
        Fx += 1000.0 * m * frx
        Fy += 1000.0 * m * fry
        Fz += 1000.0 * m * frz
        damping = 100.0 if v2 > 0.0 else 1.0
        fvx, fvy, fvz = _quat_rotate(qinv, -v0, -v1, -v2)
        Fx += damping * m * fvx
        Fy += damping * m * fvy
        Fz += damping * m * fvz
        sign_q = 1.0 if qinv[0] >= 0.0 else -1.0
        ix, iy, iz = _matvec3(I, sign_q * qinv[1], sign_q * qinv[2], sign_q * qinv[3])
        Mx += 1000.0 * ix
        My += 1000.0 * iy
        Mz += 1000.0 * iz
        dx, dy, dz = _matvec3(I, -wb0, -wb1, -wb2)
        Mx += 100.0 * dx
        My += 100.0 * dy
        Mz += 100.0 * dz
    if external:
        tfx, tfy, tfz = _quat_rotate(qinv, FextI[0], FextI[1], FextI[2])
        Fx += tfx
        Fy += tfy
        Fz += tfz
        Mx += MextB[0]
        My += MextB[1]
        Mz += MextB[2]
    if (
        drag_linear[0] != 0.0 or drag_linear[1] != 0.0 or drag_linear[2] != 0.0
        or drag_quadratic[0] != 0.0 or drag_quadratic[1] != 0.0 or drag_quadratic[2] != 0.0
    ):
        vbx, vby, vbz = _quat_rotate(qinv, v0, v1, v2)
        Fx += -m * (drag_linear[0] + drag_quadratic[0] * abs(vbx)) * vbx
        Fy += -m * (drag_linear[1] + drag_quadratic[1] * abs(vby)) * vby
        Fz += -m * (drag_linear[2] + drag_quadratic[2] * abs(vbz)) * vbz
    iw0, iw1, iw2 = _matvec3(I, wb0, wb1, wb2)
    wd0, wd1, wd2 = _matvec3(Iinv, Mx - (wb1 * iw2 - wb2 * iw1), My - (wb2 * iw0 - wb0 * iw2), Mz - (wb0 * iw1 - wb1 * iw0))
    ds[n] = wd0
    ds[n + 1] = wd1
    ds[n + 2] = wd2
    hx = 0.5 * wb0
    hy = 0.5 * wb1
    hz = 0.5 * wb2
    qw = s[n + 3]
    qx = s[n + 4]
    qy = s[n + 5]
    qz = s[n + 6]
    ds[n + 3] = -hx * qx - hy * qy - hz * qz
    ds[n + 4] = hx * qw + hz * qy - hy * qz
    ds[n + 5] = hy * qw - hz * qx + hx * qz
    ds[n + 6] = hz * qw + hy * qx - hx * qy
    fb0 = Fx / m
    fb1 = Fy / m
    fb2 = Fz / m
    a0, a1, a2 = _quat_rotate(q, fb0, fb1, fb2)
    ds[n + 7] = a0
    ds[n + 8] = a1
    ds[n + 9] = a2 + gravity
    ds[n + 10] = v0
    ds[n + 11] = v1
    ds[n + 12] = v2
    fspB[0] = fb0
    fspB[1] = fb1
    fspB[2] = fb2
    wDotB[0] = wd0
    wDotB[1] = wd1
    wDotB[2] = wd2
    return ds


@njit(cache=True)
def _multirotor_tick(
    inputs, rotor_velocity, wB, q, vI, xI, fspB, wDotB, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm, rotor_tau,
    rotor_wmax, rotor_kesc, rotor_izz, m, I, Iinv, FextI, MextB, external, drag_linear, drag_quadratic, gravity, dt,
    substeps,
):
    # Classical Runge-Kutta steps over the tick with the motor commands held; fspB and wDotB, which the IMU reads, at
    # its end.
    n = rotor_velocity.shape[0]
    s = np.concatenate((rotor_velocity, wB, q, vI, xI))
    h = dt / substeps
    for _ in range(substeps):
        k1 = _multirotor_derivative(s, inputs, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm, rotor_tau, rotor_wmax,
                                    rotor_kesc, rotor_izz, m, I, Iinv, FextI, MextB, external, drag_linear,
                                    drag_quadratic, gravity, fspB, wDotB)
        k2 = _multirotor_derivative(s + 0.5 * h * k1, inputs, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm,
                                    rotor_tau, rotor_wmax, rotor_kesc, rotor_izz, m, I, Iinv, FextI, MextB, external,
                                    drag_linear, drag_quadratic, gravity, fspB, wDotB)
        k3 = _multirotor_derivative(s + 0.5 * h * k2, inputs, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm,
                                    rotor_tau, rotor_wmax, rotor_kesc, rotor_izz, m, I, Iinv, FextI, MextB, external,
                                    drag_linear, drag_quadratic, gravity, fspB, wDotB)
        k4 = _multirotor_derivative(s + h * k3, inputs, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm, rotor_tau,
                                    rotor_wmax, rotor_kesc, rotor_izz, m, I, Iinv, FextI, MextB, external, drag_linear,
                                    drag_quadratic, gravity, fspB, wDotB)
        s += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        s[n + 3:n + 7] /= np.sqrt(s[n + 3] ** 2 + s[n + 4] ** 2 + s[n + 5] ** 2 + s[n + 6] ** 2)
    rotor_velocity[:] = s[:n]
    wB[:] = s[n:n + 3]
    q[:] = s[n + 3:n + 7]
    vI[:] = s[n + 7:n + 10]
    xI[:] = s[n + 10:]
    _multirotor_derivative(s, inputs, rotor_r, rotor_axis, rotor_dir, rotor_k, rotor_cm, rotor_tau, rotor_wmax,
                           rotor_kesc, rotor_izz, m, I, Iinv, FextI, MextB, external, drag_linear, drag_quadratic,
                           gravity, fspB, wDotB)


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
        self.w = 0.

class MultiRotor:
    def __init__(self, gravity=GRAVITY):
        if not np.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("gravity must be finite and positive")
        self.rotors = []
        self.gravity = float(gravity)
        self.m = 1.
        self.I = np.eye(3, dtype=np.float32)
        self.Iinv = np.eye(3, dtype=np.float32)
        # The state is double precision: single precision's rounding of each 0.5 ms step accumulates over a flight to
        # tens of micrometres, the scale of the tracking errors flown.
        self.xI = np.array([0., 0., 0.], dtype=np.float64)
        self.vI = np.array([0., 0., 0.], dtype=np.float64)
        self.fspB = np.array([0., 0., 0.], dtype=np.float64)
        self.q = np.array([1., 0., 0., 0.], dtype=np.float64)
        self.wDotB = np.array([0., 0., 0.], dtype=np.float64)
        self.wB = np.array([0., 0., 0.], dtype=np.float64)

        self.throw_time = +np.inf
        self.throw_duration = 0.
        self.FthrowI = np.array([0., 0., 0.], dtype=np.float32)
        self.MthrowB = np.array([0., 0., 0.], dtype=np.float32)
        self.dragLinearB = np.zeros(3, dtype=np.float32)
        self.dragQuadraticB = np.zeros(3, dtype=np.float32)
        # Runge-Kutta steps per tick: two hold a 4 s hover within 0.2 nm of sixteen.
        self.substeps = 2

    def __repr__(self):
        qWrong = np.zeros_like(self.q)
        qWrong[3] = self.q[0]
        qWrong[:3] = self.q[1:]
        rot = R.from_quat(qWrong)
        eulers = rot.as_euler('ZYX', degrees=True)
        return f"MultiRotor( n={len(self.rotors)}, x={self.xI}m, v={self.vI}m/s, roll={eulers[2]}deg, pitch={eulers[1]}deg, yaw={eulers[0]}deg )"

    def throw(self, height=3.5, acc=45., wB=[0., 0., 0.], vHorz=[0., 0.], at_time=0.):
        force = self.m * ( acc + self.gravity )

        # solve duration:
        # 
        # height = height after powered throw (sT) + altitude gained during coasting (sC)
        # sT = 0.5*a*t**2
        # vT = a*t
        # sC = 0.5*vT**2 / g,  becayse 0.5*vT**2 = g*sC
        # 
        # then, solve  height == sT + sC  for time
        self.throw_time = -at_time
        self.throw_duration = np.sqrt( 2. * height / (acc * (1. + acc / self.gravity)) )

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
        self.rotorVelocity = np.zeros(self.n, np.float64)
        self.inputs = np.zeros(self.n, np.float32)
        self._refreshRotorCache()
        # A zero-length tick compiles the integrator before the first timed one.
        self.tick(0.0)

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

    def setPose(self, x=[0., 0., 0.], q=[1., 0., 0., 0.]):
        self.xI[:] = np.asarray(x, dtype=np.float64)
        self.q[:] = np.asarray(q, dtype=np.float64)

    def setTwist(self, v=[0., 0., 0.], w=[0., 0., 0.]):
        self.vI[:] = np.asarray(v, dtype=np.float64)
        self.wB[:] = np.asarray(w, dtype=np.float64)

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
            ustar = ( 1.1 * self.gravity / np.sqrt(max(v)) ) * (Nr @ V[:, np.argmax(v)])
            if not ( ((ustar >= 0.) & (ustar <= 1.)).all() or ((ustar >= -1.) & (ustar <= 0.)).all()):
                print(f"\nWARNING: generated craft does not have enough thrust-to-weight to hover without rotation. Double check rotation directions")
                return False

        return True

    def tick(self, dt):
        self.throw_time += dt
        _multirotor_tick(
            self.inputs, self.rotorVelocity, self.wB, self.q, self.vI, self.xI, self.fspB, self.wDotB, self._rotor_r,
            self._rotor_axis, self._rotor_dir, self._rotor_k, self._rotor_cm, self._rotor_tau, self._rotor_wmax,
            self._rotor_kesc, self._rotor_izz, self.m, self.I, self.Iinv, self.FthrowI, self.MthrowB,
            0. < self.throw_time <= self.throw_duration, self.dragLinearB, self.dragQuadraticB, self.gravity, dt,
            self.substeps,
        )
        for i, rotor in enumerate(self.rotors):
            rotor.w = float(self.rotorVelocity[i])
            rotor.F[:] = rotor.axis * rotor.k * rotor.w * rotor.w

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
        self.update()

    def update(self):
        _imu_update_no_noise(
            self.uav.fspB,
            self.uav.wDotB,
            self.uav.wB,
            self.r,
            self.qInv,
            self.acc,
            self.gyro,
        )
        if self.accStd != 0.0 or self.gyroStd != 0.0:
            self.acc += np.random.normal(self.accBias, self.accStd)
            self.gyro += np.random.normal(self.gyroBias, self.gyroStd)
