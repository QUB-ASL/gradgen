import React, {useEffect} from 'react';
import type {ReactNode} from 'react';
import {useLocation} from '@docusaurus/router';

declare global {
  interface Window {
    MathJax?: {
      typeset?: () => void;
      typesetPromise?: (elements?: Element[]) => Promise<void>;
    };
  }
}

function MathJaxTypesetter({children}: {children: ReactNode}) {
  const location = useLocation();

  useEffect(() => {
    const runTypeset = () => {
      if (window.MathJax?.typesetPromise) {
        void window.MathJax.typesetPromise();
        return;
      }
      if (window.MathJax?.typeset) {
        window.MathJax.typeset();
      }
    };

    // Let route content settle before typesetting.
    const timer = window.setTimeout(runTypeset, 0);
    return () => window.clearTimeout(timer);
  }, [location.pathname]);

  return <>{children}</>;
}

export default function Root({children}: {children: ReactNode}): ReactNode {
  return <MathJaxTypesetter>{children}</MathJaxTypesetter>;
}
