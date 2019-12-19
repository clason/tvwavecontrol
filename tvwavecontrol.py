"""
Solve transmission version of optimal control problem
    ½‖By-yᵈ‖² + αG(u) + βTV(u) s.t.  yₜₜ-div(u∇y) = f
with multi-bang (G) and total variation (TV) regularization
For details, see
Christian Clason, Karl Kunisch, Philip Trautmann
Optimal control of the principal coefficient in a scalar wave equation
arXiv:1912.08672
"""
from dolfin import *
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix, kron
from scipy.sparse.linalg import factorized
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


def assemble_csr(a, bc=None):
    """assemble bilinear form with boundary conditions to CSR sparse matrix"""
    A = assemble(a)
    if bc:
        bc.apply(A)
    mat = as_backend_type(A).mat()
    return csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)


def submeasure(mesh, x1, x2, y1, y2):
    """construct subdomain for (bi)linear forms"""
    class markdomain(SubDomain):
        def inside(self, x, on_boundary):
            return (between(x[0], (x1, x2)) and between(x[1], (y1, y2)))
    markfun = MeshFunction("size_t", mesh, mesh.topology().dim())
    markfun.set_all(0)
    markdomain().mark(markfun, 1)
    return Measure('dx', domain=mesh, subdomain_data=markfun)


# parameters
nx = 64     # space discretization: number of subintervals on [-1,1]
nt = 128    # time discretization: number of time intervals
ui = [0.0, 0.1, 0.2, 0.3, 0.4]  # admissible states
m = len(ui)                     # number of admissible states
umin = 1.0                      # lower bound u
alpha = 1e-5    # multibang penalty parameter
beta = 1e-4     # TV penalty parameter
delta = 0.1     # relative noise level (wrt max norm)
tol = 1e-6      # tolerance for relative residual norm

# mesh, function space, test and trial functions in time
T = 3.0
dt = T/nt
tgrid = IntervalMesh(nt, 0.0, T)
VT = FunctionSpace(tgrid, "Lagrange", 1)
ut = TrialFunction(VT)
vt = TestFunction(VT)
Mt = assemble_csr(ut*vt*dx)  # mass matrix in time
sigma = 0.25                 # Zlotnik parameter

# mesh, function space, trial- and test functions in space
mesh = RectangleMesh(Point(-1, -1), Point(1, 2), nx, nx)
V = FunctionSpace(mesh, "Lagrange", 1)
nx = V.dim()
v = TestFunction(V)
w = TrialFunction(V)
u_fe = Function(V)        # place holder for iterate
M = assemble_csr(w*v*dx)  # mass matrix in Omega

# source: Diracs in space, Ricker wavelet in time
g = Function(V)
G = g.vector()
for i in range(-9, 9):
    PointSource(V, Point(i/10, -0.9), 1.0).apply(G)
    PointSource(V, Point(0.05+i/10, -0.8), 1.0).apply(G)
G = G[:]
f = Expression('A*(1.0-2.0*pow(pi*f*(x[0]-t0),2))*exp(-pow(pi*f*(x[0]-t0),2))',
               f=5.0, A=2.0, t0=0.1, degree=2)
f = interpolate(f, VT)
F = Mt*np.flipud(f.vector())

# control and observation domain
dc = submeasure(mesh, -1.0, 1.0, 0.0, 1.0)
do = submeasure(mesh, -1.0, 1.0, 1.0, 2.0)
di = assemble(w*dx)[:]  # vector of element sizes
Mo = assemble_csr(w*v*do(1))
Mtx = kron(Mt, Mo)  # space-time mass matrix in observation domain

# gradient for TV
Psi = FunctionSpace(mesh, 'DG', 0)
D1 = assemble_csr(Dx(w, 0)*TestFunction(Psi)*dc(1))
D2 = assemble_csr(Dx(w, 1)*TestFunction(Psi)*dc(1))
nel = Psi.dim()

# plotting
X = interpolate(Expression("x[0]", degree=1), V).vector().get_local()
Y = interpolate(Expression("x[1]", degree=1), V).vector().get_local()


def plot_fun(Z, title, fig=None):
    """plot function"""
    if fig is None:
        fig = plt.figure(figsize=(10, 5))

    fig.clear()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_trisurf(X, Y, Z,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_ticks(ui)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
    ax.set_title(title)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_trisurf(X, Y, Z,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_ticks(ui)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
    ax.view_init(azim=270, elev=0)

    fig.tight_layout()
    plt.draw()
    plt.pause(.001)
    return fig


def mb_prox(q, tau):
    """evaluate proximal mapping for multibang penalty"""
    at = alpha*tau/2
    w = (q < ((1+at)*ui[0]+at*ui[1]))*ui[0]
    for i in range(1, m):
        w += (q >= ((1+at)*ui[i-1]+at*ui[i])) * \
            (q < ((1+at)*ui[i]+at*ui[i-1]))*(q-at*(ui[i]+ui[i-1]))
        if i != 1:
            w += (q >= ((1+at)*ui[i-1]+at*ui[i-2])) * \
                (q < ((1+at)*ui[i-1]+at*ui[i]))*ui[i-1]
    w += (q >= ((1+at)*ui[m-1]+at*ui[m-2]))*ui[m-1]
    return w


def tv_proj(psi1, psi2):
    """project on ell2-Linfty ball, overwriting psi!"""
    if beta == 0:
        psi1[:] = 0.0
        psi2[:] = 0.0
    else:
        psi_mag = np.maximum(beta, np.sqrt(psi1**2 + psi2**2))/beta
        psi1 /= psi_mag
        psi2 /= psi_mag
    return psi1, psi2


def forward(A, AMinv):
    """solve forward wave equation with prefactorized stiffness matrix"""
    Y = np.zeros([nx, nt+1])
    # zero initial state
    y0 = Y[:, 0]
    # first step
    b = dt*F[0]*G
    y1 = AMinv(b)
    Y[:, 1] = y1
    # time stepping
    t = dt
    i = 1
    while (t < T):
        h = 2.0*y1-y0
        b = dt*F[i]*G - dt*dt*A*y1
        y0 = y1
        y1 = AMinv(b) + h
        i += 1
        Y[:, i] = y1
        t += dt
    return Y


def backward(A, AMinv, r):
    """solve backward wave equation with prefactorized stiffness matrix"""
    # integrate r in time
    r = (Mo*r)*Mt
    P = np.zeros([nx, nt+1])
    # zero end time
    pT = P[:, nt]
    # first step
    b = dt*r[:, nt]
    p1 = AMinv(b)
    P[:, nt-1] = p1
    # time stepping
    t = T - dt
    i = nt-1
    while (t > 0):
        h = 2.0*p1-pT
        b = dt*r[:, i] - dt*dt*A*p1
        pT = p1
        p1 = AMinv(b) + h
        i -= 1
        P[:, i] = p1
        t -= dt

    return P


def gradient(A, AMinv, P):
    """evaluate the gradient of tracking including forward time-stepping"""
    gradS = np.zeros(nx)
    y = Function(V)
    # zero initial state
    y0 = np.zeros(nx)
    # first step
    b = dt*F[0]*G
    y1 = AMinv(b)
    t = dt
    i = 1
    # time stepping
    while (t <= T):
        y.vector().set_local(y1)
        Fg = assemble_csr(inner(grad(y), grad(w))*v*dc(1))
        if (i == nt):
            gradS += sigma*(Fg*P[:, nt-1])
        else:
            gradS += Fg*(sigma*P[:, i-1] + (1-2*sigma)
                         * P[:, i] + sigma*P[:, i+1])
            h = 2.0*y1-y0
            b = dt*F[i]*G - dt*dt*A*y1
            y0 = y1
            y1 = AMinv(b) + h
        i += 1
        t += dt
    gradS *= dt
    return gradS


def S(u):
    """evaluate control-to-state mapping"""
    u_fe.vector()[:] = u
    A = assemble_csr(inner(u_fe*grad(w), grad(v))*dc(1) +
                     inner(umin*grad(w), grad(v))*dx)
    AMinv = factorized(csc_matrix(M+sigma*dt*dt*A))
    y = forward(A, AMinv)
    return y


def Sp(u, r):
    """evaluate adjoint derivative of control-to-state mapping"""
    u_fe.vector()[:] = u
    A = assemble_csr(inner(u_fe*grad(w), grad(v))*dc(1) +
                     inner(umin*grad(w), grad(v))*dx)
    AMinv = factorized(csc_matrix(M+sigma*dt*dt*A))
    P = backward(A, AMinv, r)
    z = gradient(A, AMinv, P)
    return z


def pdps(u, yd, psi1, psi2):
    """implement the primal-dual proximal splitting algorithm"""
    ub = u.copy()
    uold = u.copy()
    r = yd - S(u)

    sig1 = 1e-1  # step size for S', tbd
    sig2 = 1e-1  # step size for Dh, tbd
    tau = 1e3  # step size for primal, tbd
    maxit = 3001

    fig = None
    for k in range(maxit):
        uold[:] = u
        u -= tau*(Sp(u, r)+D1.T*psi1+D2.T*psi2)
        u[:] = mb_prox(u, tau)

        if k % 10 == 0:
            fig = plot_fun(u, "iteration %d" % (k), fig)
            # compute residuals
            resu = np.sum(di*(uold-u)**2)
            pres = sig1/(sig1+1)*(r-(yd-S(uold))).flatten('F')
            resr = pres.dot(Mtx*pres)
            psk1 = psi1 + sig2*(D1*uold)
            psk2 = psi2 + sig2*(D2*uold)
            psk1, psk2 = tv_proj(psk1, psk2)
            resp = np.linalg.norm(psi1-psk1)**2 + np.linalg.norm(psi2-psk2)**2
            resn = sqrt(resu + resr + resp)
            print('It:', k, ' res:', resn)
            if resn < tol:
                break

        ub[:] = 2*u-uold
        r += sig1*(yd-S(ub))
        r /= (1+sig1)

        psi1 += sig2*(D1*ub)
        psi2 += sig2*(D2*ub)
        psi1, psi2 = tv_proj(psi1, psi2)

    return u


# exact parameter
ue = Expression('''u1*((x[1]>=0)&&(x[1]<=1))+
                   (u3-u1)*((x[0]<+0.5)&&(x[0]>=-0.0)&&(x[1]>=0.0)&&(x[1]<=0.5))+
                   (u4-u1)*((x[0]<-0.0)&&(x[0]>=-0.5)&&(x[1]>=0.0)&&(x[1]<=0.5))+
                   (u2-u1)*((x[0]<-0.5)&&(x[0]>=-1.0)&&(x[1]>=0.0)&&(x[1]<=0.5))+
                   (u4-u1)*((x[0]<+0.5)&&(x[0]>=-0.0)&&(x[1]>0.5)&&(x[1]<=1))+
                   (u3-u1)*((x[0]<-0.0)&&(x[0]>=-0.5)&&(x[1]>0.5)&&(x[1]<=1))+
                   (u2-u1)*((x[0]<-0.5)&&(x[0]>=-1.0)&&(x[1]>0.5)&&(x[1]<=1))
                ''', degree=0, u1=ui[1], u2=ui[2], u3=ui[3], u4=ui[4])
ue = interpolate(ue, V)
plot_fun(ue.vector(), "exact coefficient")

# exact, noisy observation
np.random.seed(632423)
ye = S(ue.vector()[:])
yd = ye + delta*np.random.randn(nx, nt+1)*np.max(np.abs(ye))

# starting values for u and psi
u = np.zeros(nx)
psi1 = np.zeros(nel)
psi2 = np.zeros(nel)

# run algorithm
u = pdps(u, yd, psi1, psi2)
