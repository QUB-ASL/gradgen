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
        self.__name = 'the_uncertain_gradiator'

    def __create_gradients(self):
        """Create Jacobian of functions of ellx(w), ellu(w), fx(w), fu(w), vfx, and turn them into instances of class
        """
        for index_w in self.__w_list:
            self.__jfx_list[index_w] = cs.jacobian(self.__f_list[index_w], self.__x).T @ self.__d
            self.__jfu_list[index_w] = cs.jacobian(self.__f_list[index_w], self.__u).T @ self.__d
            self.__ellx_list[index_w] = cs.jacobian(self.__ell_list[index_w], self.__x).T
            self.__ellu_list[index_w] = cs.jacobian(self.__ell_list[index_w], self.__u).T

        self.__vfx = cs.jacobian(self.__vf, self.__x).T

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
        c_interface_rendered = c_interface_template.render(name=self.__name,
                                                           w=self.__w_list)
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
            'lib_stochastic.rs', subdir='casadi-rs')
        casadi_lib_rs_rendered = casadi_lib_rs_template.render(
            name=self.__name,
            nx=self.__nx,
            nu=self.__nu,
            N=self.__N,
            w=self.__w_list)
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
