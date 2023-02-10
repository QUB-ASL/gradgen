import casadi.casadi as cs
import subprocess as subp
import os
import shutil
import jinja2
from gradgen.definitions import *


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
        self.__x = x
        self.__u = u
        self.__f = f
        self.__ell = ell
        self.__vf = vf
        self.__N = N
        self.__nx = x.size()[0]
        self.__nu = u.size()[0]
        self.__d = cs.SX.sym('d', self.__nx)
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
        self.__name = 'gradgenz'
        self.__destination_path = 'codegenz'

    def __target_root_dir(self):
        trgt_root_abspath = os.path.join(self.__destination_path, self.__name)
        return os.path.abspath(trgt_root_abspath)

    def __target_externc_dir(self):
        dest_abspath = os.path.join(
            self.__destination_path, self.__name, 'casadi_'+self.__name, 'extern')
        return os.path.abspath(dest_abspath)

    def __target_casadirs_dir(self):
        casadirs_abspath = os.path.join(
            self.__destination_path, self.__name, 'casadi_'+self.__name)
        return os.path.abspath(casadirs_abspath)

    def __create_dirs(self):
        if not os.path.exists(self.__target_externc_dir()):
            os.makedirs(self.__target_externc_dir())
        casadi_src_path = os.path.join(
            self.__destination_path, self.__name, 'casadi_'+self.__name, 'src')
        main_src_path = os.path.join(
            self.__destination_path, self.__name, 'src')
        if not os.path.exists(casadi_src_path):
            os.makedirs(casadi_src_path)
        if not os.path.exists(main_src_path):
            os.makedirs(main_src_path)

    @staticmethod
    def __get_template(name, subdir=None):
        subdir_path = templates_subdir(subdir)
        file_loader = jinja2.FileSystemLoader(subdir_path)
        env = jinja2.Environment(loader=file_loader, autoescape=True)
        return env.get_template(name)

    def with_name(self, name):
        self.__name = name
        return self

    def with_target_path(self, dst_path):
        self.__destination_path = dst_path
        return self

    def __create_gradients(self):
        self.__jfx = cs.jacobian(self.__f, self.__x).T @ self.__d
        self.__jfu = cs.jacobian(self.__f, self.__u).T @ self.__d
        self.__ellx = cs.jacobian(self.__ell, self.__x).T
        self.__ellu = cs.jacobian(self.__ell, self.__u).T
        self.__vfx = cs.jacobian(self.__vf, self.__x).T

    def __function_name(self, fname):
        return 'casadi_' + self.__name + '_' + fname

    def __generate_casadi_functions(self):
        self.__f_fun = cs.Function(self.__function_name('f'), [self.__x, self.__u], [
            self.__f], ['x', 'u'], ['f'])
        self.__jfx_fun = cs.Function(self.__function_name(
            'jfx'), [self.__x, self.__u, self.__d], [self.__jfx], ['x', 'u', 'd'], ['jfx'])
        self.__jfu_fun = cs.Function(self.__function_name(
            'jfu'), [self.__x, self.__u, self.__d], [self.__jfu], ['x', 'u', 'd'], ['jfu'])
        self.__ell_fun = cs.Function(self.__function_name(
            'ell'), [self.__x, self.__u], [self.__ell], ['x', 'u'], ['ell'])
        self.__ellx_fun = cs.Function(self.__function_name('ellx'), [self.__x, self.__u], [
            self.__ellx], ['x', 'u'], ['ellx'])
        self.__ellu_fun = cs.Function(self.__function_name('ellu'), [self.__x, self.__u], [
            self.__ellu], ['x', 'u'], ['ellu'])
        self.__vf_fun = cs.Function(self.__function_name(
            'vf'), [self.__x], [self.__vf], ['x'], ['vf'])
        self.__vfx_fun = cs.Function(self.__function_name(
            'vfx'), [self.__x], [self.__vfx], ['x'], ['vfx'])

    def __generate_c_code(self):
        c_code_filename = 'casadi_functions.c'
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
        target_dir = self.__target_externc_dir()
        shutil.move(c_code_filename, os.path.join(target_dir, c_code_filename))

    def __generate_glob_header(self):
        global_header_template = CostGradient.__get_template(
            'global_header.h.tmpl', subdir='c')
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
            self.__target_externc_dir(), "glob_header.h")
        with open(glob_header_target_path, "w") as fh:
            fh.write(global_header_rendered)

    def __generate_c_interface(self):
        c_interface_template = CostGradient.__get_template(
            'autograd_interface.c.tmpl', subdir='c')
        c_interface_rendered = c_interface_template.render(name=self.__name)
        c_interface_target_path = os.path.join(
            self.__target_externc_dir(), "interface.c")
        with open(c_interface_target_path, "w") as fh:
            fh.write(c_interface_rendered)

    def __prepare_casadi_rs(self):
        # Cargo.toml [casadi]
        cargo_template = CostGradient.__get_template(
            'Cargo.toml', subdir='casadi-rs')
        cargo_rendered = cargo_template.render(name=self.__name)
        cargo_target_path = os.path.join(
            self.__target_casadirs_dir(), "Cargo.toml")
        with open(cargo_target_path, "w") as fh:
            fh.write(cargo_rendered)
        # build.rs
        build_rs_template = CostGradient.__get_template(
            'build.rs', subdir='casadi-rs')
        build_rs_rendered = build_rs_template.render(name=self.__name)
        build_rs_target_path = os.path.join(
            self.__target_casadirs_dir(), "build.rs")
        with open(build_rs_target_path, "w") as fh:
            fh.write(build_rs_rendered)
        # lib.rs
        casadi_lib_rs_template = CostGradient.__get_template(
            'lib.rs', subdir='casadi-rs')
        casadi_lib_rs_rendered = casadi_lib_rs_template.render(
            name=self.__name,
            nx=self.__nx,
            nu=self.__nu,
            N=self.__N)
        casadi_lib_rs_target_path = os.path.join(
            self.__target_casadirs_dir(), "src", "lib.rs")
        with open(casadi_lib_rs_target_path, "w") as fh:
            fh.write(casadi_lib_rs_rendered)

    def __generate_rust_lib(self):
        # Cargo
        cargo_template = CostGradient.__get_template(
            'Cargo.toml', subdir='rust')
        cargo_rendered = cargo_template.render(name=self.__name)
        cargo_target_path = os.path.join(
            self.__target_root_dir(), "Cargo.toml")
        with open(cargo_target_path, "w") as fh:
            fh.write(cargo_rendered)
        # lib
        lib_template = CostGradient.__get_template('lib.rs', subdir='rust')
        lib_rendered = lib_template.render(name=self.__name)
        lib_target_path = os.path.join(
            self.__target_root_dir(), "src", "lib.rs")
        with open(lib_target_path, "w") as fh:
            fh.write(lib_rendered)

    def __cargo_build(self):
        cmd = ['cargo', 'build', '-q']
        p = subp.Popen(cmd, cwd=self.__target_root_dir())
        process_completion = p.wait()
        if process_completion != 0:
            raise Exception('Rust build failed')

    def build(self, no_rust_build=False):
        self.__create_dirs()
        self.__create_gradients()
        self.__generate_casadi_functions()
        self.__generate_c_code()
        self.__generate_glob_header()
        self.__generate_c_interface()
        self.__prepare_casadi_rs()
        self.__generate_rust_lib()
        if not no_rust_build:
            self.__cargo_build()
