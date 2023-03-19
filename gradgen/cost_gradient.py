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
        :param N: prediction horizon
        """
        self.__x = x
        self.__u = u
        self.__f = f
        self.__ell = ell
        self.__vf = vf
        self.__N = N
        self.__nx = self.__x.size()[0]
        self.__nu = self.__u.size()[0]
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
        self.__name = 'gradgen'
        self.__destination_path = 'codegenz'

    def __target_root_dir(self):
        """Import destination path and name path and concatenate the paths to form a new complete target root path

        :return: an absolute version of string which represents the concatenated path components target root path
        """
        trgt_root_abspath = os.path.join(self.__destination_path, self.__name)
        return os.path.abspath(trgt_root_abspath)

    def __target_externc_dir(self):
        """Import destination path, name path, casadi_name path and extern path concatenate the paths to form a new complete target destination path

        :return: an absolute version of string which represents the concatenated path components target external path
        """
        dest_abspath = os.path.join(
            self.__destination_path, self.__name, 'casadi_'+self.__name, 'extern')
        return os.path.abspath(dest_abspath)

    def __target_casadirs_dir(self):
        """Import destination path, name path and casadi_name path concatenate the paths to form a new complete target casadi path

        :return: an absolute version of string which represents the concatenated path components target casadi path
        """
        casadirs_abspath = os.path.join(
            self.__destination_path, self.__name, 'casadi_'+self.__name)
        return os.path.abspath(casadirs_abspath)

    def __create_dirs(self):
        """If there is no target external path exist, make a target external path;
           Import destination path, name path, casadi_name path and src path concatenate the paths to form a new complete casadi src path;
           Import destination path, name path and src path concatenate the paths to form a new complete mian src path;
           If there is not casadi src path exist, make a casadi src path;
           If there is not mian src path exist, make a mian src path;
        """

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
        """Use the template Environment.
        Use instances of this class to store the configuration and global objects,
        and load templates from the file system or other locations by name with loader
        and return a template.

        :param name: Receive user's input name. It is the name of the template to load.
        :param subdir: There is no subdirectories of named directories for processing, defaults to None
        :return: load a template from the environment by name with loader and return a Template.
        """
        subdir_path = templates_subdir(subdir)
        file_loader = jinja2.FileSystemLoader(subdir_path)
        env = jinja2.Environment(loader=file_loader, autoescape=True)
        return env.get_template(name)

    def with_name(self, name):
        """Turn name into instances of class

        :param name: take user input as a name and assign it to the name associated with the object
        :return: an instance of the class
        """
        self.__name = name
        return self

    def with_target_path(self, dst_path):
        """Turn final destination for generated code into instances of class

        :param dst_path: final destination for generated code
        :return: an instance of the class
        """
        self.__destination_path = dst_path
        return self

    def __create_gradients(self):
        """Create Jacobian of function ellx, ellu, fx, fu and turn them into instances of class
        """
        self.__jfx = cs.jacobian(self.__f, self.__x).T @ self.__d
        self.__jfu = cs.jacobian(self.__f, self.__u).T @ self.__d
        self.__ellx = cs.jacobian(self.__ell, self.__x).T
        self.__ellu = cs.jacobian(self.__ell, self.__u).T
        self.__vfx = cs.jacobian(self.__vf, self.__x).T

    def __function_name(self, fname):
        """Create function name

        :param fname: receive function name from user
        :return: full casadi gradgen function name
        """
        return 'casadi_' + self.__name + '_' + fname

    def __generate_casadi_functions(self):
        """Create casadi functions and turn them into instances of class
        """
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
        """Generate C code for casadi functions
        """
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
        """Generate global header:
        Load a template from this environment, return the loaded global header template
        and render it with some variables to generate global header render.
        And join target externc path and glob_header.h together.
        Then open the file from above path in a write mode
        and then writing to it replaces the existing content.
        """
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
        """Generate C interface:
        Load a template from this environment, return the c interface template
        and render it with name variable to generate c interface render.
        And join target externc path and interface.c together.
        Then open the file from above path in a write mode
        and then writing to it replaces the existing content.
        """
        c_interface_template = CostGradient.__get_template(
            'autograd_interface.c.tmpl', subdir='c')
        c_interface_rendered = c_interface_template.render(name=self.__name)
        c_interface_target_path = os.path.join(
            self.__target_externc_dir(), "interface.c")
        with open(c_interface_target_path, "w") as fh:
            fh.write(c_interface_rendered)

    def __prepare_casadi_rs(self):
        """Prepare casadi:
        Load a template from this environment, return the cargo, build, casadi library template
        and render it with variables to generate render.
        And join target casadi path, Cargo.toml, build.rs, lib.rs, src together.
        Then open the file from above path in a write mode
        and then writing to it replaces the existing content.
        """
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
        """Generate Rust library:
        Load a template from this environment, return the cargo, library template
        and render it with variables to generate render.
        And join target root path, Cargo.toml, lib.rs, src together.
        Then open the file from above path in a write mode
        and then writing to it replaces the existing content.
        """
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
        """Build cargo
        Spawn the child process and override the current working directory with target root dictionary
        and check whether the cargo build is successful

        :raises Exception: Rust build may fail
        """
        cmd = ['cargo', 'build', '-q']
        p = subp.Popen(cmd, cwd=self.__target_root_dir())
        process_completion = p.wait()
        if process_completion != 0:
            raise Exception('Rust build failed')

    def build(self, no_rust_build=False):
        """Build all the function we need to calculate gradient

        :param no_rust_build: If set to True, the code will be generated, but it will not be compiled, defaults to False
        """
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
