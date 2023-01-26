import casadi.casadi as cs
import subprocess as subp


class CostGradient:

    def __init__(self, x, u, f, ell, vf, N):
        # TODO make all these attributes private (later)
        # TODO write docs
        self.x = x
        self.u = u
        self.f = f
        self.ell = ell
        self.vf = vf
        self.N = N
        self.nx = x.size()[0]
        self.nu = u.size()[0]
        self.dx = cs.SX.sym('dx', self.nx)
        self.du = cs.SX.sym('du', self.nu)
        self.jfx = None
        self.jfu = None
        self.ellx = None
        self.ellu = None
        self.vfx = None
        self.f_fun = None
        self.jfx_fun = None
        self.jfu_fun = None
        self.ell_fun = None
        self.ellx_fun = None
        self.ellu_fun = None
        self.vf_fun = None
        self.vfx_fun = None
        self.__c_code_dir = '.'
        self.__c_codename = 'autogen_jacobians'
        self.__c_compiler = 'gcc'
        self.__c_compiler_flags = ['-fPIC']

    def with_c_code_destination(self, c_code_dir):
        self.__c_code_dir = c_code_dir
        return self

    def with_c_compiler(self, c_compiler):
        self.__c_compiler = c_compiler
        return self

    def with_extra_c_compiler_flags(self, extra_c_flags):
        self.__c_compiler_flags += extra_c_flags
        return self

    def create_gradients(self):
        self.jfx = cs.jacobian(self.f, self.x) @ self.dx
        self.jfu = cs.jacobian(self.f, self.u) @ self.du
        self.ellx = cs.jacobian(self.ell, self.x).T
        self.ellu = cs.jacobian(self.ell, self.u).T
        self.vfx = cs.jacobian(self.vf, self.x).T

    def generate_casadi_functions(self):
        self.f_fun = cs.Function('dynamics', [self.x, self.u], [
                                 self.f])
        self.jfx_fun = cs.Function(
            'jfx', [self.x, self.u, self.dx], [self.jfx], ['x', 'u', 'dx'], ['jfx'])
        self.jfu_fun = cs.Function('jfu', [self.x, self.u, self.du], [
                                   self.jfu], ['x', 'u', 'du'], ['jfu'])
        self.ell_fun = cs.Function('ell', [self.x, self.u], [
                                   self.ell], ['x', 'u'], ['ell'])
        self.ellx_fun = cs.Function('ellx', [self.x, self.u], [
                                    self.ellx], ['x', 'u'], ['ellx'])
        self.ellu_fun = cs.Function('ellu', [self.x, self.u], [
                                    self.ellu], ['x', 'u'], ['ellu'])
        self.vf_fun = cs.Function(
            'vf', [self.x], [self.vf], ['x'], ['vf'])
        self.vfx_fun = cs.Function(
            'vfx', [self.x], [self.vfx], ['x'], ['vfx'])

    def generate_c_code(self):
        codegen = cs.CodeGenerator(self.__c_codename + '.c')
        codegen.add(self.f_fun)
        codegen.add(self.jfx_fun)
        codegen.add(self.jfu_fun)
        codegen.add(self.ell_fun)
        codegen.add(self.ellx_fun)
        codegen.add(self.ellu_fun)
        codegen.add(self.vf_fun)
        codegen.add(self.vfx_fun)
        codegen.generate()

    def compile_c_code(self):
        compilation_cmd = [self.__c_compiler] + self.__c_compiler_flags + \
            ['-shared', self.__c_codename + '.c', '-o', self.__c_codename + '.so']
        out = subp.run(compilation_cmd)
