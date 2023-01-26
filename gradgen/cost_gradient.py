import casadi.casadi as cs
import subprocess as subp


class CostGradient:

    def __init__(self, x, u, f, ell, vf, N):
        # TODO write docs
        self.__x = x
        self.__u = u
        self.__f = f
        self.__ell = ell
        self.__vf = vf
        self.__N = N
        self.__nx = x.size()[0]
        self.__nu = u.size()[0]
        self.__dx = cs.SX.sym('dx', self.__nx)
        self.__du = cs.SX.sym('du', self.__nu)
        self.__jfx = None
        self.__jfu = None
        self.__ellx = None
        self.__ellu = None
        self.__vfx = None
        self.__f_fun = None
        self.__jfx_fun = None
        self.__jfu_fun = None
        self.__ell_fun = None
        self.__ellx_fun = None
        self.__ellu_fun = None
        self.__vf_fun = None
        self.__vfx_fun = None
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

    def __create_gradients(self):
        self.__jfx = cs.jacobian(self.__f, self.__x) @ self.__dx
        self.__jfu = cs.jacobian(self.__f, self.__u) @ self.__du
        self.__ellx = cs.jacobian(self.__ell, self.__x).T
        self.__ellu = cs.jacobian(self.__ell, self.__u).T
        self.__vfx = cs.jacobian(self.__vf, self.__x).T

    def __generate_casadi_functions(self):
        self.__f_fun = cs.Function('dynamics', [self.__x, self.__u], [
            self.__f])
        self.__jfx_fun = cs.Function(
            'jfx', [self.__x, self.__u, self.__dx], [self.__jfx], ['x', 'u', 'dx'], ['jfx'])
        self.__jfu_fun = cs.Function('jfu', [self.__x, self.__u, self.__du], [
            self.__jfu], ['x', 'u', 'du'], ['jfu'])
        self.__ell_fun = cs.Function('ell', [self.__x, self.__u], [
            self.__ell], ['x', 'u'], ['ell'])
        self.__ellx_fun = cs.Function('ellx', [self.__x, self.__u], [
            self.__ellx], ['x', 'u'], ['ellx'])
        self.__ellu_fun = cs.Function('ellu', [self.__x, self.__u], [
            self.__ellu], ['x', 'u'], ['ellu'])
        self.__vf_fun = cs.Function(
            'vf', [self.__x], [self.__vf], ['x'], ['vf'])
        self.__vfx_fun = cs.Function(
            'vfx', [self.__x], [self.__vfx], ['x'], ['vfx'])

    def __generate_c_code(self):
        codegen = cs.CodeGenerator(self.__c_codename + '.c')
        codegen.add(self.__f_fun)
        codegen.add(self.__jfx_fun)
        codegen.add(self.__jfu_fun)
        codegen.add(self.__ell_fun)
        codegen.add(self.__ellx_fun)
        codegen.add(self.__ellu_fun)
        codegen.add(self.__vf_fun)
        codegen.add(self.__vfx_fun)
        codegen.generate()

    def __compile_c_code(self):
        compilation_cmd = [self.__c_compiler] + self.__c_compiler_flags + \
            ['-shared', self.__c_codename + '.c', '-o', self.__c_codename + '.so']
        out = subp.run(compilation_cmd)

    def build(self):
        self.__create_gradients()
        self.__generate_casadi_functions()
        self.__generate_c_code()
        self.__compile_c_code()
