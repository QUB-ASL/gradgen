import casadi.casadi as cs
import subprocess as subp
import os
import shutil
import jinja2
import gradgen.definitions as dfn


class CostGradient:

    def __init__(self, x, u, f, ell, vf, N):
        """
        Create new CostGradient object

        :param x: state symbol
        :param u: input symbol
        :param f: system dynamics symbol (depends on x, u)
        :param ell: cost function symbol (depends on x, u)
        :param vf: terminal cost symbol (depends on x)
        :param N: prediction horizon (int) 
        """
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
        self.__name = 'autogen_cost_grad'
        self.__destination_path = 'codegenz'
        self.__c_compiler = 'gcc'
        self.__c_compiler_flags = ['-fPIC']
        self.__prepare()

    def __get_target_code_dir_asbpath(self):
        dest_abspath = os.path.join(self.__destination_path, self.__name)
        return os.path.abspath(dest_abspath)

    def __prepare(self):
        if not os.path.exists(self.__get_target_code_dir_asbpath()):
            os.makedirs(self.__get_target_code_dir_asbpath())

    @staticmethod
    def __get_template(name, subdir=None):
        subdir_path = dfn.templates_subdir(subdir)
        file_loader = jinja2.FileSystemLoader(subdir_path)
        env = jinja2.Environment(loader=file_loader, autoescape=True)
        return env.get_template(name)

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
        self.__f_fun = cs.Function(self.__name+'_f', [self.__x, self.__u], [
            self.__f], ['x', 'u'], ['f'])
        self.__jfx_fun = cs.Function(
            self.__name+'jfx', [self.__x, self.__u, self.__dx], [self.__jfx], ['x', 'u', 'dx'], ['jfx'])
        self.__jfu_fun = cs.Function(self.__name+'_jfu', [self.__x, self.__u, self.__du], [
            self.__jfu], ['x', 'u', 'du'], ['jfu'])
        self.__ell_fun = cs.Function(self.__name+'_ell', [self.__x, self.__u], [
            self.__ell], ['x', 'u'], ['ell'])
        self.__ellx_fun = cs.Function(self.__name+'_ellx', [self.__x, self.__u], [
            self.__ellx], ['x', 'u'], ['ellx'])
        self.__ellu_fun = cs.Function(self.__name+'_ellu', [self.__x, self.__u], [
            self.__ellu], ['x', 'u'], ['ellu'])
        self.__vf_fun = cs.Function(
            self.__name+'_vf', [self.__x], [self.__vf], ['x'], ['vf'])
        self.__vfx_fun = cs.Function(
            self.__name+'_vfx', [self.__x], [self.__vfx], ['x'], ['vfx'])

    def __generate_c_code(self):
        c_code_filename = self.__name + '.c'
        codegen = cs.CodeGenerator(c_code_filename)
        codegen.add(self.__f_fun)
        codegen.add(self.__jfx_fun)
        codegen.add(self.__jfu_fun)
        codegen.add(self.__ell_fun)
        codegen.add(self.__ellx_fun)
        codegen.add(self.__ellu_fun)
        codegen.add(self.__vf_fun)
        codegen.add(self.__vfx_fun)
        codegen.generate()
        # Move generated C code to destination directory
        target_dir = self.__get_target_code_dir_asbpath()
        shutil.move(c_code_filename, os.path.join(target_dir, c_code_filename))

    def __generate_glob_header(self):
        global_header_template = CostGradient.__get_template(
            'global_header.h.tmpl', subdir='autograd')
        global_header_rendered = global_header_template.render(
            name=self.__name,
            f=self.__f_fun,
            jfx=self.__jfx_fun,
            jfu=self.__jfu_fun,
            ell=self.__ell_fun,
            ellx=self.__ellx_fun,
            ellu=self.__ellu_fun,
            vf=self.__vf_fun,
            vfx=self.__vfx_fun,
            N=self.__N,
            nx=self.__nx,
            nu=self.__nu
        )
        glob_header_target_path = os.path.join(
            self.__get_target_code_dir_asbpath(), "glob_header.h")
        with open(glob_header_target_path, "w") as fh:
            fh.write(global_header_rendered)

    def __generate_c_interface(self):
        c_interface_template = CostGradient.__get_template(
            'autograd_interface.c.tmpl', subdir='autograd')
        c_interface_rendered = c_interface_template.render(name=self.__name)
        c_interface_target_path = os.path.join(
            self.__get_target_code_dir_asbpath(), "interface.c")
        with open(c_interface_target_path, "w") as fh:
            fh.write(c_interface_rendered)

    def build(self):
        self.__create_gradients()
        self.__generate_casadi_functions()
        self.__generate_c_code()
        self.__generate_glob_header()
        self.__generate_c_interface()
