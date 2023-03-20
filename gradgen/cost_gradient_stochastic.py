import casadi.casadi as cs
import subprocess as subp
import os
import shutil
import jinja2
from gradgen.definitions import *
from gradgen.cost_gradient import CostGradient


class CostGradientStochastic(CostGradient):

    def __init__(self, tree, x, u, f, ell, vf):
        """
        Create new stochastic CostGradient object
        :param tree: scenario tree for stochastic ocp
        :param x: state symbol
        :param u: input symbol
        :param w: list of possible events
        :param f: list of system dynamics symbol (depends on x, u, w)
        :param ell: list of cost function symbol (depends on x, u, w)
        :param vf: terminal cost symbol (depends on x)
        """
        self.__tree = tree
        super().__init__(x, u, None, None, None, None)
        self.__x = x
        self.__u = u
        self.__w = cs.SX.sym('w', 1)
        self.__f_list = f
        self.__ell_list = ell
        self.__nx = self.__x.size()[0]
        self.__nu = self.__u.size()[0]
        self.__nw = len(self.__f_list)
        if self.__nw != len(self.__ell_list):
            raise Exception(
                "number of dynamics functions does not equal number of cost functions")
        self.__d = cs.SX.sym('d', self.__nx)
        self.__N = self.__tree.num_stages - 1
        self.__f_selector = None
        self.__jfx = None
        self.__jfu = None
        self.__ell_selector = None
        self.__ellx = None
        self.__ellu = None
        self.__f_fun = None
        self.__jfx_fun = None
        self.__jfu_fun = None
        self.__ell_fun = None
        self.__ellx_fun = None
        self.__ellu_fun = None
        self.__vf = vf

    def __create_gradients(self):
        """Create Jacobian of functions of ellx(w), ellu(w), fx(w), fu(w), vfx, and turn them into instances of class
        """
        self.__f_selector = cs.if_else(
            cs.fabs(self.__w - (self.__nw - 1)) <= 0.1, self.__f_list[-1], cs.SX_nan(1))
        self.__ell_selector = cs.if_else(
            cs.fabs(self.__w - (self.__nw - 1)) <= 0.1, self.__ell_list[-1], cs.SX_nan(1))
        for i_events in range(self.__nw - 2, -1, -1):
            self.__f_selector = cs.if_else(
                cs.fabs(self.__w - i_events) <= 0.1, self.__f_list[i_events], self.__f_selector)
            self.__ell_selector = cs.if_else(
                cs.fabs(self.__w - i_events) <= 0.1, self.__ell_list[i_events], self.__ell_selector)

        self.__jfx = cs.jacobian(self.__f_selector, self.__x).T @ self.__d
        self.__jfu = cs.jacobian(self.__f_selector, self.__u).T @ self.__d
        self.__ellx = cs.jacobian(self.__ell_selector, self.__x).T
        self.__ellu = cs.jacobian(self.__ell_selector, self.__u).T
        self.__vfx = cs.jacobian(self.__vf, self.__x).T

    def __generate_casadi_functions(self):
        """Create casadi functions and turn them into instances of class
        """
        self.__f_fun = cs.Function(self._CostGradient__function_name('f'),
                                   [self.__x, self.__u, self.__w],
                                   [self.__f_selector],
                                   ['x', 'u', 'w'],
                                   ['f'])
        self.__jfx_fun = cs.Function(self._CostGradient__function_name('jfx'),
                                     [self.__x, self.__u, self.__d, self.__w],
                                     [self.__jfx],
                                     ['x', 'u', 'd', 'w'],
                                     ['jfx'])
        self.__jfu_fun = cs.Function(self._CostGradient__function_name('jfu'),
                                     [self.__x, self.__u, self.__d, self.__w],
                                     [self.__jfu],
                                     ['x', 'u', 'd', 'w'],
                                     ['jfu'])
        self.__ell_fun = cs.Function(self._CostGradient__function_name('ell'),
                                     [self.__x, self.__u, self.__w],
                                     [self.__ell_selector],
                                     ['x', 'u', 'w'],
                                     ['ell'])
        self.__ellx_fun = cs.Function(self._CostGradient__function_name('ellx'),
                                      [self.__x, self.__u, self.__w],
                                      [self.__ellx],
                                      ['x', 'u', 'w'],
                                      ['ellx'])
        self.__ellu_fun = cs.Function(self._CostGradient__function_name('ellu'),
                                      [self.__x, self.__u, self.__w],
                                      [self.__ellu],
                                      ['x', 'u', 'w'],
                                      ['ellu'])
        self.__vf_fun = cs.Function(self._CostGradient__function_name('vf'),
                                    [self.__x],
                                    [self.__vf],
                                    ['x'],
                                    ['vf'])
        self.__vfx_fun = cs.Function(self._CostGradient__function_name('vfx'),
                                     [self.__x],
                                     [self.__vfx],
                                     ['x'],
                                     ['vfx'])

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
        target_dir = self._CostGradient__target_externc_dir()
        shutil.move(c_code_filename, os.path.join(target_dir, c_code_filename))

    def __generate_glob_header(self):
        """Generate global header:
        Load a template from this environment, return the loaded global header template
        and render it with some variables to generate global header render.
        And join target externc path and glob_header.h together.
        Then open the file from above path in a write mode
        and then writing to it replaces the existing content.
        """
        global_header_template = self._CostGradient__get_template(
            'global_header_stochastic.h.tmpl', subdir='c')

        global_header_rendered = global_header_template.render(
            name=self.name(),
            nx=self.__nx,
            nu=self.__nu,
            N=self.__N,
            f=self.__f_fun,
            jfx=self.__jfx_fun,
            jfu=self.__jfu_fun,
            ell=self.__ell_fun,
            ellx=self.__ellx_fun,
            ellu=self.__ellu_fun,
            vf=self.__vf_fun,
            vfx=self.__vfx_fun
        )
        glob_header_target_path = os.path.join(
            self._CostGradient__target_externc_dir(), "glob_header.h")
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
        c_interface_template = self._CostGradient__get_template(
            'autograd_interface_stochastic.c.tmpl', subdir='c')
        c_interface_rendered = c_interface_template.render(name=self.name())
        c_interface_target_path = os.path.join(
            self._CostGradient__target_externc_dir(), "interface.c")
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

        children = self.__tree._ScenarioTree__children
        children_from = [x[0] for x in children]
        children_to = [x[-1] for x in children]

        nodes_at_stage_from = [0]
        nodes_at_stage_to = [0]
        for i in range(1, self.__tree.num_stages):
            nodes_at_stage_i = self.__tree.nodes_at_stage(i)
            nodes_at_stage_from += [nodes_at_stage_i[0]]
            nodes_at_stage_to += [nodes_at_stage_i[-1]]

        # number of different w's
        num_ws = len(set(self.__tree._ScenarioTree__w_idx[1:]))

        # Cargo.toml [casadi]
        cargo_template = self._CostGradient__get_template(
            'Cargo.toml', subdir='casadi-rs')
        cargo_rendered = cargo_template.render(name=self.name())
        cargo_target_path = os.path.join(
            self._CostGradient__target_casadirs_dir(), "Cargo.toml")
        with open(cargo_target_path, "w") as fh:
            fh.write(cargo_rendered)
        # build.rs
        build_rs_template = self._CostGradient__get_template(
            'build.rs', subdir='casadi-rs')
        build_rs_rendered = build_rs_template.render(name=self.name())
        build_rs_target_path = os.path.join(
            self._CostGradient__target_casadirs_dir(), "build.rs")
        with open(build_rs_target_path, "w") as fh:
            fh.write(build_rs_rendered)
        # lib.rs
        casadi_lib_rs_template = self._CostGradient__get_template(
            'lib_stochastic.rs', subdir='casadi-rs')
        casadi_lib_rs_rendered = casadi_lib_rs_template.render(
            name=self.name(),
            nx=self.__nx,
            nu=self.__nu,
            N=self.__N,
            tree=self.__tree,
            children_from=children_from,
            children_to=children_to,
            nodes_at_stage_from=nodes_at_stage_from,
            nodes_at_stage_to=nodes_at_stage_to,
            num_events=num_ws)
        casadi_lib_rs_target_path = os.path.join(
            self._CostGradient__target_casadirs_dir(), "src", "lib.rs")
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
        cargo_template = self._CostGradient__get_template(
            'Cargo.toml', subdir='rust')
        cargo_rendered = cargo_template.render(name=self.name())
        cargo_target_path = os.path.join(
            self._CostGradient__target_root_dir(), "Cargo.toml")
        with open(cargo_target_path, "w") as fh:
            fh.write(cargo_rendered)
        # lib
        lib_template = self._CostGradient__get_template(
            'lib.rs', subdir='rust')
        lib_rendered = lib_template.render(name=self.name())
        lib_target_path = os.path.join(
            self._CostGradient__target_root_dir(), "src", "lib.rs")
        with open(lib_target_path, "w") as fh:
            fh.write(lib_rendered)

    def build(self, no_rust_build=False):
        """Build all the function we need to calculate gradient

        :param no_rust_build: If set to True, the code will be generated, but it will not be compiled, defaults to False
        """
        self._CostGradient__create_dirs()
        self.__create_gradients()
        self.__generate_casadi_functions()
        self.__generate_c_code()
        self.__generate_glob_header()
        self.__generate_c_interface()
        self.__prepare_casadi_rs()
        self.__generate_rust_lib()
        if not no_rust_build:
            self.__cargo_build()
