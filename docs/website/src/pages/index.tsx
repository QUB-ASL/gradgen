import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const featureCards = [
  {
    title: 'Symbolic expressions',
    body:
      'Build symbolic expressions and functions in Python',
  },
  {
    title: 'Embeddable code',
    body:
      'Gradgen generates no_std rust crates',
  },
  {
    title: 'Optional Python bridge',
    body:
      'When enabled, Gradgen emits a sibling PyO3 package so generated crates can be imported directly from Python.',
  },
];

function Hero(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <section className={styles.hero}>
      <div className="container">
        <div className={styles.heroGrid}>
          <div className={styles.heroCopy}>
            <div className={styles.kicker}>
              Symbolic differentiation: Python to Rust</div>
            <Heading as="h1" className={styles.title}>
              Generate Rust crates kernels from Python
            </Heading>
            <p className={styles.subtitle}>{siteConfig.tagline}.</p>
            {/* <p className={styles.description}>
              Gradgen turns symbolic functions into Rust crates,
              suitable for embedded applications.
            </p> */}

            <div className={styles.buttons}>
              <Link className="button button--primary button--lg" to="/docs/intro">
                Start reading
              </Link>
              <Link className="button button--secondary button--lg" to="/docs/category/demos">
                Browse demos
              </Link>
            </div>

            <div className={styles.metaRow}>
              <span>Rust codegen</span>
              <span>Embeddable</span>
              <span>Fast + Efficient</span>
            </div>
          </div>

          <div className={styles.heroPanel}>
            <div className={styles.panelLabel}>Quick start</div>
            <pre className={styles.codeBlock}>
              <code>{`from gradgen import Function, SX

x = SX.sym("x", 2)
f = Function("energy", 
             [x], [x.norm2sq()], 
             input_names=["x"], 
             output_names=["cost"])

project = f.create_rust_project("./my_crates")`}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

function FeatureCards(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeading}>
          <Heading as="h2">What it does</Heading>
          <p>
            Using gradgen you can create symbolic variables and expressions, define functions, perform automatic differentiation, and generate embedable (<code>#![no_std]</code>) Rust code for your functions and their gradients, Hessians, and/or Hessian-vector products.
          </p>
          <figure className={styles.sectionImage}>
            <img
              src="/gradgen/img/gradgen-what-it-does.png"
              alt="Gradgen workflow illustration"
              className={styles.sectionImageImg}
            />
          </figure>
        </div>
        
        <p>
            
        </p>

        
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={siteConfig.title}
      description="Gradgen generates Rust crates from symbolic Python functions, including derivatives and optional Python wrappers.">
      <Hero />
      <main>
        <FeatureCards />
      </main>
    </Layout>
  );
}
