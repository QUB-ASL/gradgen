import casadi.casadi as cs
import subprocess as subp
import os
import shutil
import jinja2
from gradgen.definitions import *
from gradgen.cost_gradient import CostGradient


class CostGradientStochastic(CostGradient):

    def __init__(self, tree, x, u, w, f, ell, vf):
        """
        Create new stochastic CostGradient object
        :param tree: scenario tree for stochastic ocp
        :param x: state symbol
        :param u: input symbol
        :param w: list of events
        :param f: list of system dynamics symbol (depends on x, u, w)
        :param ell: list of cost function symbol (depends on x, u, w)
        :param vf: terminal cost symbol (depends on x)
        """
        self.__tree = tree
        super().__init__(x, u, None, None, None, None)
        self.__x = x
        self.__u = u
        self.__w_list = w
        self.__f_list = f
        self.__ell_list = ell
        self.__nx = self.__x.size()[0]
        self.__nu = self.__u.size()[0]
        self.__d = cs.SX.sym('d', self.__nx)
        self.__N = self.__tree.num_stages - 1
        self.__nw = len(self.__w_list)
        self.__jfx_list = [None] * self.__nw
        self.__jfu_list = [None] * self.__nw
        self.__ellx_list = [None] * self.__nw
        self.__ellu_list = [None] * self.__nw
        self.__f_fun_list = [None] * self.__nw
        self.__jfx_fun_list = [None] * self.__nw
        self.__jfu_fun_list = [None] * self.__nw
        self.__ell_fun_list = [None] * self.__nw
        self.__ellx_fun_list = [None] * self.__nw
        self.__ellu_fun_list = [None] * self.__nw
        self.__vf = vf
        self.__name = 'the uncertain gradiator'

    # def __target_root_dir(self):
    #     """Import destination path and name path and concatenate the paths to form a new complete target root path
    #
    #     :return: an absolute version of string which represents the concatenated path components target root path
    #     """
    #     trgt_root_abspath = os.path.join(self.__destination_path, self.__name)
    #     return os.path.abspath(trgt_root_abspath)

    # def __target_externc_dir(self):
    #     """Import destination path, name path, casadi_name path and extern path.
    #     Concatenate the paths to form a new complete target destination path.
    #
    #     :return: an absolute version of string which represents the concatenated path components target external path
    #     """
    #     dest_abspath = os.path.join(
    #         self.__destination_path, self.__name, 'casadi_' + self.__name, 'extern')
    #     return os.path.abspath(dest_abspath)

    # def __target_casadirs_dir(self):
    #     """Import destination path, name path and casadi_name path.
    #     Concatenate the paths to form a new complete target casadi path.
    #
    #     :return: an absolute version of string which represents the concatenated path components target casadi path
    #     """
    #     casadirs_abspath = os.path.join(
    #         self.__destination_path, self.__name, 'casadi_' + self.__name)
    #     return os.path.abspath(casadirs_abspath)

    # def __create_dirs(self):
    #     """If there is no target external path exist, make a target external path;
    #        Import destination path, name path, casadi_name path and src path.
    #        Concatenate the paths to form a new complete casadi src path;
    #        Import destination path, name path and src path concatenate the paths to form a new complete mian src path;
    #        If there is no casadi src path exist, make a casadi src path;
    #        If there is no main src path exist, make a mian src path;
    #     """
    #     if not os.path.exists(self.__target_externc_dir()):
    #         os.makedirs(self.__target_externc_dir())
    #     casadi_src_path = os.path.join(
    #         self.__destination_path, self.__name, 'casadi_' + self.__name, 'src')
    #     main_src_path = os.path.join(
    #         self.__destination_path, self.__name, 'src')
    #     if not os.path.exists(casadi_src_path):
    #         os.makedirs(casadi_src_path)
    #     if not os.path.exists(main_src_path):
    #         os.makedirs(main_src_path)

    # @staticmethod
    # def __get_template(name, subdir=None):
    #     """Use the template Environment.
    #     Use instances of this class to store the configuration and global objects,
    #     and load templates from the file system or other locations by name with loader
    #     and return a template.
    #
    #     :param name: Receive user's input name. It is the name of the template to load.
    #     :param subdir: There is no subdirectories of named directories for processing, defaults to None
    #     :return: load a template from the environment by name with loader and return a Template.
    #     """
    #     subdir_path = templates_subdir(subdir)
    #     file_loader = jinja2.FileSystemLoader(subdir_path)
    #     env = jinja2.Environment(loader=file_loader, autoescape=True)
    #     return env.get_template(name)

    # def with_name(self, name):
    #     """Turn name into instances of class
    #
    #     :param name: take user input as a name and assign it to the name associated with the object
    #     :return: an instance of the class
    #     """
    #     self.__name = name
    #     return self

    # def with_target_path(self, dst_path):
    #     """Turn final destination for generated code into instances of class
    #
    #     :param dst_path: final destination for generated code
    #     :return: an instance of the class
    #     """
    #     self.__destination_path = dst_path
    #     return self

    def __create_gradients(self):
        """Create Jacobian of functions of ellx(w), ellu(w), fx(w), fu(w), vfx, and turn them into instances of class
        """
        for index_w in self.__w_list:
            self.__jfx_list[index_w] = cs.jacobian(self.__f_list[index_w], self.__x).T @ self.__d
            self.__jfu_list[index_w] = cs.jacobian(self.__f_list[index_w], self.__u).T @ self.__d
            self.__ellx_list[index_w] = cs.jacobian(self.__ell_list[index_w], self.__x).T
            self.__ellu_list[index_w] = cs.jacobian(self.__ell_list[index_w], self.__u).T

        self.__vfx = cs.jacobian(self.__vf, self.__x).T

    # def __function_name(self, fname):
    #     """Create function name
    #
    #     :param fname: receive function name from user
    #     :return: full casadi gradgen function name
    #     """
    #     return 'casadi_' + self.__name + '_' + fname

    def __generate_casadi_functions(self):
        """Create casadi functions and turn them into instances of class
        """
        for index_w in self.__w_list:
            self.__f_fun_list[index_w] = cs.Function(self._function_name(f"f_{index_w}"),
                                                     [self.__x, self.__u],
                                                     [self.__f_list[index_w]],
                                                     ['x', 'u'],
                                                     [f"f_{index_w}"])
            self.__jfx_fun_list[index_w] = cs.Function(self._function_name(f"jfx_{index_w}"),
                                                       [self.__x, self.__u, self.__d],
                                                       [self.__jfx_list[index_w]],
                                                       ['x', 'u', 'd'],
                                                       [f"jfx_{index_w}"])
            self.__jfu_fun_list[index_w] = cs.Function(self._function_name(f"jfu_{index_w}"),
                                                       [self.__x, self.__u, self.__d],
                                                       [self.__jfu_list[index_w]],
                                                       ['x', 'u', 'd'],
                                                       [f"jfu_{index_w}"])
            self.__ell_fun_list[index_w] = cs.Function(self._function_name(f"ell_{index_w}"),
                                                       [self.__x, self.__u],
                                                       [self.__ell_list[index_w]],
                                                       ['x', 'u'],
                                                       [f"ell_{index_w}"])
            self.__ellx_fun_list[index_w] = cs.Function(self._function_name(f"ellx_{index_w}"),
                                                        [self.__x, self.__u],
                                                        [self.__ellx_list[index_w]],
                                                        ['x', 'u'],
                                                        [f"ellx_{index_w}"])
            self.__ellu_fun_list[index_w] = cs.Function(self._function_name(f"ellu_{index_w}"),
                                                        [self.__x, self.__u],
                                                        [self.__ellu_list[index_w]],
                                                        ['x', 'u'],
                                                        [f"ellu_{index_w}"])

        self.__vf_fun = cs.Function(self._function_name(
            'vf'), [self.__x], [self.__vf], ['x'], ['vf'])
        self.__vfx_fun = cs.Function(self._function_name(
            'vfx'), [self.__x], [self.__vfx], ['x'], ['vfx'])

    def __generate_c_code(self):
        """Generate C code for casadi functions
        """
        c_code_filename = 'casadi_functions.c'
        codegen = cs.CodeGenerator(c_code_filename)
        for index_w in self.__w_list:
            codegen.add(self.__f_fun_list[index_w])
            codegen.add(self.__jfx_fun_list[index_w])
            codegen.add(self.__jfu_fun_list[index_w])
            codegen.add(self.__ell_fun_list[index_w])
            codegen.add(self.__ellx_fun_list[index_w])
            codegen.add(self.__ellu_fun_list[index_w])

        codegen.add(self.__vf_fun)
        codegen.add(self.__vfx_fun)
        codegen.generate()
        # Move generated C code to destination directory
        target_dir = self._target_externc_dir()
        shutil.move(c_code_filename, os.path.join(target_dir, c_code_filename))

    def __generate_glob_header(self):
        """Generate global header:
        Load a template from this environment, return the loaded global header template
        and render it with some variables to generate global header render.
        And join target externc path and glob_header.h together.
        Then open the file from above path in a write mode
        and then writing to it replaces the existing content.
        """
        global_header_template = CostGradient._get_template(
            'global_header_stochastic.h.tmpl', subdir='c')
        global_header_rendered = global_header_template.render(
            name=self.__name,
            nx=self.__nx,
            nu=self.__nu,
            w=self.__w_list,
            N=self.__N,
            f_list=self.__f_fun_list,
            jfx_list=self.__jfx_fun_list,
            jfu_list=self.__jfu_fun_list,
            ell_list=self.__ell_fun_list,
            ellx_list=self.__ellx_fun_list,
            ellu_list=self.__ellu_fun_list,
            vf=self.__vf_fun,
            vfx=self.__vfx_fun
        )
        glob_header_target_path = os.path.join(
            self._target_externc_dir(), "glob_header.h")
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
        c_interface_template = CostGradient._get_template(
            'autograd_interface_stochastic.c.tmpl', subdir='c')
        c_interface_rendered = c_interface_template.render(name=self.__name)
        c_interface_target_path = os.path.join(
            self._target_externc_dir(), "interface.c")
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
        cargo_template = CostGradient._get_template(
            'Cargo.toml', subdir='casadi-rs')
        cargo_rendered = cargo_template.render(name=self.__name)
        cargo_target_path = os.path.join(
            self._target_casadirs_dir(), "Cargo.toml")
        with open(cargo_target_path, "w") as fh:
            fh.write(cargo_rendered)
        # build.rs
        build_rs_template = CostGradient._get_template(
            'build.rs', subdir='casadi-rs')
        build_rs_rendered = build_rs_template.render(name=self.__name)
        build_rs_target_path = os.path.join(
            self._target_casadirs_dir(), "build.rs")
        with open(build_rs_target_path, "w") as fh:
            fh.write(build_rs_rendered)
        # lib.rs
        casadi_lib_rs_template = CostGradient._get_template(
            'lib.rs', subdir='casadi-rs')
        casadi_lib_rs_rendered = casadi_lib_rs_template.render(
            name=self.__name,
            nx=self.__nx,
            nu=self.__nu,
            N=self.__N)
        casadi_lib_rs_target_path = os.path.join(
            self._target_casadirs_dir(), "src", "lib.rs")
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
        cargo_template = CostGradient._get_template(
            'Cargo.toml', subdir='rust')
        cargo_rendered = cargo_template.render(name=self.__name)
        cargo_target_path = os.path.join(
            self._target_root_dir(), "Cargo.toml")
        with open(cargo_target_path, "w") as fh:
            fh.write(cargo_rendered)
        # lib
        lib_template = CostGradient._get_template('lib.rs', subdir='rust')
        lib_rendered = lib_template.render(name=self.__name)
        lib_target_path = os.path.join(
            self._target_root_dir(), "src", "lib.rs")
        with open(lib_target_path, "w") as fh:
            fh.write(lib_rendered)

    # def __cargo_build(self):
    #     """Build cargo
    #     Spawn the child process and override the current working directory with target root dictionary
    #     and check whether the cargo build is successful
    #
    #     :raises Exception: Rust build may fail
    #     """
    #     cmd = ['cargo', 'build', '-q']
    #     p = subp.Popen(cmd, cwd=self.__target_root_dir())
    #     process_completion = p.wait()
    #     if process_completion != 0:
    #         raise Exception('Rust build failed')

    def build(self, no_rust_build=False):
        """Build all the function we need to calculate gradient

        :param no_rust_build: If set to True, the code will be generated, but it will not be compiled, defaults to False
        """
        self._create_dirs()
        self.__create_gradients()
        self.__generate_casadi_functions()
        self.__generate_c_code()
        self.__generate_glob_header()
        self.__generate_c_interface()
        self.__prepare_casadi_rs()
        self.__generate_rust_lib()
        if not no_rust_build:
            self.__cargo_build()
